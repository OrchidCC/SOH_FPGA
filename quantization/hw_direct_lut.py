#!/usr/bin/env python3
"""Direct LUT optimization: learn 16 entries per channel instead of sin+GridWarp.

Instead of: requant → sin(float) → GridWarp(scale, bias)  [2 params/ch]
We do:      requant → LUT[channel][index]                  [16 params/ch]

More expressive: 16 free parameters vs 2 (scale, bias).
Hardware-aligned: LUT is exactly what the FPGA implements.
"""

import sys, os, copy, time, math, numpy as np, torch, torch.nn as nn
from torch.autograd import Function, grad
import torch.nn.functional as F
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.Model import Solution_u, MLP, Sin
from utils.util import eval_metrix
from utils.dataloader import XJTUdata

# ─── Fake Quant ────────────────────────────────────────────────────

class FakeQuantSTE(Function):
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        return torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    @staticmethod
    def backward(ctx, go):
        return go, None, None, None

def fq_weight(w, wb):
    qm = 2**(wb-1)-1
    with torch.no_grad():
        s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qm
    return FakeQuantSTE.apply(w, s, -qm, qm)

def fq_act(h, ab=4):
    qm = 2**(ab-1)-1
    with torch.no_grad():
        s = h.abs().amax().clamp(min=1e-12) / qm
    return FakeQuantSTE.apply(h, s, -qm, qm)

# ─── Per-channel GridWarp (for H baseline) ────────────────────────

class ChannelGridWarp(nn.Module):
    def __init__(self, n_channels, a_bits=4):
        super().__init__()
        self.a_bits = a_bits
        self.log_scale = nn.Parameter(torch.zeros(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))

    def init_from_stats(self, act_tensor):
        with torch.no_grad():
            scale = torch.quantile(act_tensor.abs(), 0.995, dim=0).clamp(min=0.05)
            self.log_scale.copy_(torch.log(scale))
            self.bias.copy_(act_tensor.mean(dim=0))

    def forward(self, x):
        scale = torch.exp(self.log_scale)
        centered = (x - self.bias) / scale
        qm = 2**(self.a_bits-1)-1
        q = FakeQuantSTE.apply(centered, torch.tensor(1.0/qm, device=x.device), -qm, qm)
        return q * scale + self.bias

# ─── H model: pre-sin quant + GridWarp ──────────────────────────

class HWAlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.predictor = nn.Module()
        self.encoder.net = nn.Sequential(
            nn.Linear(17, 60), Sin(),
            nn.Linear(60, 60), Sin(), nn.Dropout(0.2),
            nn.Linear(60, 32),
        )
        self.predictor.net = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(32, 32), Sin(), nn.Linear(32, 1),
        )
        self.act_warps = nn.ModuleList([
            ChannelGridWarp(60, 4), ChannelGridWarp(60, 4), ChannelGridWarp(32, 4),
        ])

    def init_warps(self, fp_model, x_calib):
        h = x_calib.detach(); si = 0
        for mod in list(fp_model.encoder.net) + list(fp_model.predictor.net):
            if isinstance(mod, nn.Linear):
                with torch.no_grad(): h = mod(h)
            elif isinstance(mod, Sin):
                with torch.no_grad(): h = torch.sin(h)
                if si < len(self.act_warps):
                    self.act_warps[si].init_from_stats(h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass

    def forward(self, x):
        h = x; si = 0
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                h = F.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                h = fq_act(h)
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h

# ─── L model: pre-sin quant + direct learnable LUT ──────────────

class DirectLUTModel(nn.Module):
    """Replace sin+GridWarp with a directly optimized 16-entry per-channel LUT.

    16 free parameters per channel (vs 2 for GridWarp).
    Exactly matches the FPGA implementation.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.predictor = nn.Module()
        self.encoder.net = nn.Sequential(
            nn.Linear(17, 60), Sin(),
            nn.Linear(60, 60), Sin(), nn.Dropout(0.2),
            nn.Linear(60, 32),
        )
        self.predictor.net = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(32, 32), Sin(), nn.Linear(32, 1),
        )
        # Learnable LUTs: 16 entries per channel
        self.act_luts = nn.ParameterList([
            nn.Parameter(torch.zeros(60, 16)),  # sin1: 60 channels
            nn.Parameter(torch.zeros(60, 16)),  # sin2: 60 channels
            nn.Parameter(torch.zeros(32, 16)),  # sin3: 32 channels
        ])

    def init_luts(self, fp_model, x_calib):
        """Initialize LUT entries from sin(calibration_data)."""
        h = x_calib.detach(); si = 0
        for mod in list(fp_model.encoder.net) + list(fp_model.predictor.net):
            if isinstance(mod, nn.Linear):
                with torch.no_grad(): h = mod(h)
            elif isinstance(mod, Sin):
                if si < len(self.act_luts):
                    n_ch = self.act_luts[si].shape[0]
                    qm = 7
                    # Per-channel pre-sin scale from calibration data
                    s = h.abs().amax(dim=0).clamp(min=1e-12) / qm  # (n_ch,)
                    with torch.no_grad():
                        for v in range(-8, 8):
                            v_float = v * s  # (n_ch,) per-channel float values
                            self.act_luts[si].data[:, v + 8] = torch.sin(v_float)
                with torch.no_grad(): h = torch.sin(h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass

    def forward(self, x):
        h = x; si = 0
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                h = F.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                # Pre-sin A4 quant (matches FPGA requant)
                qm = 7
                with torch.no_grad():
                    s = h.abs().amax().clamp(min=1e-12) / qm
                h_q = FakeQuantSTE.apply(h, s, -qm, qm)

                # Integer index for LUT lookup
                with torch.no_grad():
                    idx = torch.round(h_q / s).long().clamp(-8, 7) + 8  # [0,15]

                # Differentiable LUT lookup (gradient flows to LUT entries)
                lut = self.act_luts[si]  # (n_ch, 16)
                h = torch.gather(
                    lut.unsqueeze(0).expand(h.shape[0], -1, -1),  # (B, C, 16)
                    dim=2,
                    index=idx.unsqueeze(-1)  # (B, C, 1)
                ).squeeze(-1)  # (B, C)

                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h

# ─── Training ────────────────────────────────────────────────────

def train_qat(model, fnet, trainloader, epochs=80, lr=3e-4,
              lut_lr=3e-3, pinn_alpha=0.7, pinn_beta=0.2):
    lut_params, warp_params, weight_params = [], [], []
    for name, p in model.named_parameters():
        if 'act_lut' in name:
            lut_params.append(p)
        elif 'warp' in name:
            warp_params.append(p)
        else:
            weight_params.append(p)

    param_groups = [{'params': weight_params, 'lr': lr}]
    if warp_params:
        param_groups.append({'params': warp_params, 'lr': 1e-3})
    if lut_params:
        param_groups.append({'params': lut_params, 'lr': lut_lr})
    opt = torch.optim.Adam(param_groups)

    def lr_lambda(ep):
        if ep < 5: return (ep+1)/5
        return 0.5*(1+math.cos(math.pi*(ep-5)/(epochs-5)))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    relu = nn.ReLU(); lf = nn.MSELoss(); fnet.eval()
    best_loss = float('inf'); best_state = None
    for ep in range(epochs):
        model.train(); total = 0; n = 0
        for x1, x2, y1, y2 in trainloader:
            x1.requires_grad_(True); x2.requires_grad_(True)
            u1_s = model(x1); u2_s = model(x2)
            u1t_s = grad(u1_s.sum(), x1[:,-1:], create_graph=True, allow_unused=True)[0]
            u1x_s = grad(u1_s.sum(), x1[:,:-1], create_graph=True, allow_unused=True)[0]
            if u1t_s is None: u1t_s = torch.zeros_like(u1_s)
            if u1x_s is None: u1x_s = torch.zeros_like(x1[:,:-1])
            u2t_s = grad(u2_s.sum(), x2[:,-1:], create_graph=True, allow_unused=True)[0]
            u2x_s = grad(u2_s.sum(), x2[:,:-1], create_graph=True, allow_unused=True)[0]
            if u2t_s is None: u2t_s = torch.zeros_like(u2_s)
            if u2x_s is None: u2x_s = torch.zeros_like(x2[:,:-1])
            loss_data = 0.5*lf(u1_s, y1) + 0.5*lf(u2_s, y2)
            with torch.no_grad():
                F1 = fnet(torch.cat([x1.detach(), u1_s.detach(), u1x_s.detach(), u1t_s.detach()], 1))
                F2 = fnet(torch.cat([x2.detach(), u2_s.detach(), u2x_s.detach(), u2t_s.detach()], 1))
            loss_pde = 0.5*lf(u1t_s-F1, torch.zeros_like(F1)) + 0.5*lf(u2t_s-F2, torch.zeros_like(F2))
            loss_phys = relu(torch.mul(u2_s-u1_s, y1-y2)).sum()
            loss = loss_data + pinn_alpha*loss_pde + pinn_beta*loss_phys
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); n += 1
        sch.step()
        if total/n < best_loss:
            best_loss = total/n; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model

# ─── Eval ─────────────────────────────────────────────────────────

def eval_model(model, testloader):
    model.eval(); tl, pl = [], []
    with torch.no_grad():
        for x,_,y,_ in testloader:
            u = model(x); tl.append(y.numpy()); pl.append(u.numpy())
    return eval_metrix(np.concatenate(pl), np.concatenate(tl))[1]*100

# ─── Data ─────────────────────────────────────────────────────────

def load_xjtu(bn):
    args = argparse.Namespace(data='XJTU', batch=bn, batch_size=256,
                              normalization_method='min-max', log_dir=None, save_folder=None)
    root = 'data/XJTU'; data = XJTUdata(root=root, args=args)
    tl, el = [], []
    for f in os.listdir(root):
        if bn in f:
            p = os.path.join(root, f)
            if '4' in f or '8' in f: el.append(p)
            else: tl.append(p)
    tr = data.read_all(specific_path_list=tl)
    te = data.read_all(specific_path_list=el)
    return {'train': tr['train_2'], 'test': te['test_3']}

# ─── Main ─────────────────────────────────────────────────────────

def main():
    bn = '2C'
    print(f"Loading XJTU_{bn}...")
    dl = load_xjtu(bn)
    model_dir = 'models/checkpoints/XJTU/0-0'
    results = []

    for e in range(1, 11):
        mp = os.path.join(model_dir, f'Experiment{e}', 'model.pth')
        if not os.path.exists(mp): continue

        print(f"\n--- Run {e}/10 ---")
        t0 = time.time()

        ckpt = torch.load(mp, map_location='cpu', weights_only=False)
        teacher = Solution_u()
        teacher.load_state_dict(ckpt['solution_u']); teacher.eval()
        fnet = MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60, droupout=0.2)
        fnet.load_state_dict(ckpt['dynamical_F']); fnet.eval()

        fp32 = eval_model(teacher, dl['test'])
        all_x = torch.cat([x for x, _, _, _ in dl['train']])

        # H: pre-sin quant + GridWarp (2 params/ch)
        m_h = HWAlignedModel()
        m_h.load_state_dict(ckpt['solution_u'], strict=False)
        m_h.init_warps(teacher, all_x[:256])
        m_h = train_qat(m_h, fnet, dl['train'])
        h_mape = eval_model(m_h, dl['test'])

        # L: pre-sin quant + direct LUT (16 params/ch)
        m_l = DirectLUTModel()
        m_l.load_state_dict(ckpt['solution_u'], strict=False)
        m_l.init_luts(teacher, all_x[:256])
        m_l = train_qat(m_l, fnet, dl['train'])
        l_mape = eval_model(m_l, dl['test'])

        dt = time.time() - t0
        results.append({'fp32': fp32, 'H': h_mape, 'L': l_mape})
        print(f"  FP32={fp32:.2f}%  H(GridWarp)={h_mape:.2f}%  L(DirectLUT)={l_mape:.2f}%  "
              f"delta={l_mape - h_mape:+.2f}%  ({dt:.0f}s)")

    print(f"\n{'='*60}")
    print(f"XJTU_{bn} — {len(results)} runs (all hardware-aligned with pre-sin A4)")
    print(f"{'='*60}")
    for key in ['fp32', 'H', 'L']:
        vals = [r[key] for r in results]
        label = {'fp32': 'FP32', 'H': 'H: requant+sin+GridWarp (2p/ch)',
                 'L': 'L: requant+DirectLUT (16p/ch)'}[key]
        print(f"  {label:38s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    print(f"\n  L - H (direct LUT improvement): {np.mean([r['L']-r['H'] for r in results]):+.3f}")


if __name__ == '__main__':
    main()
