#!/usr/bin/env python3
"""Hardware-aligned QAT: add pre-sin activation quantization to match FPGA.

FPGA dataflow:  W4_linear → requant(>>4) → per_channel_LUT → next_layer
Software QAT:   W4_linear → sin(FP32) → GridWarp(A4)        → next_layer
                               ↑ GAP: FPGA quantizes BEFORE sin, software doesn't

Fix: add A4 fake quant before sin in QAT training.

Compare on XJTU_2C:
  P:  original QAT (no pre-sin quant) — software upper bound
  H:  hardware-aligned QAT (pre-sin A4 + sin + GridWarp)
  A:  absorption (absorb GridWarp into weights)
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
    """Per-tensor symmetric activation fake quant (simulates FPGA requant)."""
    qm = 2**(ab-1)-1
    with torch.no_grad():
        s = h.abs().amax().clamp(min=1e-12) / qm
    return FakeQuantSTE.apply(h, s, -qm, qm)

# ─── Per-channel GridWarp ────────────────────────────────────────

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

# ─── Original model (no pre-sin quant) ──────────────────────────

class OriginalModel(nn.Module):
    """P: W4 weights + sin(FP32) + per-channel GridWarp."""
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
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h

# ─── Hardware-aligned model (pre-sin quant) ──────────────────────

class HWAlignedModel(nn.Module):
    """H: W4 weights + pre-sin A4 + sin(quantized) + per-channel GridWarp.

    Matches FPGA: linear → requant(A4) → LUT(sin+GridWarp) → next_layer.
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
                h = fq_act(h)        # ← KEY: pre-sin A4 quant (matches FPGA requant)
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h

# ─── Absorption ──────────────────────────────────────────────────

def absorb_gridwarp(model):
    m = copy.deepcopy(model)
    layers = list(m.encoder.net) + list(m.predictor.net)
    sin_indices = [i for i, mod in enumerate(layers) if isinstance(mod, Sin)]
    for si, sin_idx in enumerate(sin_indices):
        if si >= len(m.act_warps): break
        gw = m.act_warps[si]
        scale = torch.exp(gw.log_scale).detach()
        bias_vec = gw.bias.detach()
        next_lin = None
        for j in range(sin_idx + 1, len(layers)):
            if isinstance(layers[j], nn.Linear):
                next_lin = layers[j]; break
        if next_lin is not None:
            with torch.no_grad():
                next_lin.bias.add_(next_lin.weight @ bias_vec)
                next_lin.weight.mul_(scale.unsqueeze(0))
        with torch.no_grad():
            gw.log_scale.zero_()
            gw.bias.zero_()
    return m

# ─── Training ────────────────────────────────────────────────────

def train_qat(model, fnet, trainloader, epochs=80, lr=3e-4,
              warp_lr=1e-3, pinn_alpha=0.7, pinn_beta=0.2):
    warp_params, weight_params = [], []
    for name, p in model.named_parameters():
        if 'warp' in name: warp_params.append(p)
        else: weight_params.append(p)
    param_groups = [{'params': weight_params, 'lr': lr}]
    if warp_params:
        param_groups.append({'params': warp_params, 'lr': warp_lr})
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
    print(f"Loading XJTU_{bn} data...")
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

        # P: Original QAT (no pre-sin quant)
        m_p = OriginalModel()
        m_p.load_state_dict(ckpt['solution_u'], strict=False)
        m_p.init_warps(teacher, all_x[:256])
        m_p = train_qat(m_p, fnet, dl['train'])
        p_mape = eval_model(m_p, dl['test'])

        # H: Hardware-aligned QAT (pre-sin A4 + sin + GridWarp)
        m_h = HWAlignedModel()
        m_h.load_state_dict(ckpt['solution_u'], strict=False)
        m_h.init_warps(teacher, all_x[:256])
        m_h = train_qat(m_h, fnet, dl['train'])
        h_mape = eval_model(m_h, dl['test'])

        # A: Absorption from P model
        m_a = absorb_gridwarp(m_p)
        a_mape = eval_model(m_a, dl['test'])

        dt = time.time() - t0
        results.append({'fp32': fp32, 'P': p_mape, 'H': h_mape, 'A': a_mape})
        print(f"  FP32={fp32:.2f}%  P={p_mape:.2f}%  H(hw)={h_mape:.2f}%  A(abs)={a_mape:.2f}%  ({dt:.0f}s)")

    print(f"\n{'='*60}")
    print(f"XJTU_{bn} — {len(results)} runs")
    print(f"{'='*60}")
    for key in ['fp32', 'P', 'H', 'A']:
        vals = [r[key] for r in results]
        label = {'fp32': 'FP32', 'P': 'P (no pre-sin q)', 'H': 'H (hw-aligned)', 'A': 'A (absorption)'}[key]
        print(f"  {label:22s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    print(f"\n  H - P (pre-sin quant cost):  {np.mean([r['H']-r['P'] for r in results]):+.3f}")
    print(f"  A - P (absorption cost):     {np.mean([r['A']-r['P'] for r in results]):+.3f}")


if __name__ == '__main__':
    main()
