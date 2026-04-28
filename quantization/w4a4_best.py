#!/usr/bin/env python3
"""Best W4A4: Per-channel GridWarp + Residual Quantization for L3.

Combines the two methods that showed improvement:
  - Per-channel ActGridWarp: -0.78% on XJTU_2C (activation distribution matching)
  - Residual Quant L3: -0.08% avg (effective precision increase for bottleneck layer)

Configs:
  B:  Per-layer GridWarp + QAT (old baseline)
  P:  Per-channel GridWarp + QAT
  R:  Per-layer GridWarp + Residual Quant L3 + QAT
  PR: Per-channel GridWarp + Residual Quant L3 + QAT (combined best)
"""

import sys, os, copy, time, json, math, numpy as np, torch, torch.nn as nn
from torch.autograd import grad, Function
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.Model import Solution_u, MLP, Sin
from utils.util import eval_metrix
from utils.dataloader import XJTUdata, TJUdata, MITdata

device = 'cpu'

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

# ─── GridWarp variants ────────────────────────────────────────────

class LayerGridWarp(nn.Module):
    def __init__(self, init_scale=0.3, init_bias=0.0, a_bits=4):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(init_scale, 1e-6))))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        self.a_bits = a_bits
    def forward(self, x):
        scale = torch.exp(self.log_scale)
        centered = (x - self.bias) / scale
        qm = 2**(self.a_bits-1)-1
        q = FakeQuantSTE.apply(centered, torch.tensor(1.0/qm), -qm, qm)
        return q * scale + self.bias

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

# ─── Models ───────────────────────────────────────────────────────

class Model_B(nn.Module):
    """Per-layer GridWarp + standard W4 (old baseline)."""
    def __init__(self, act_stats):
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
            LayerGridWarp(s['scale'], s['bias'], 4) for s in act_stats
        ])
    def forward(self, x):
        h = x; si = 0
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                h = nn.functional.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h


class Model_P(nn.Module):
    """Per-channel GridWarp + standard W4."""
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
                h = nn.functional.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h


class Model_R(nn.Module):
    """Per-layer GridWarp + Residual Quant L3."""
    def __init__(self, act_stats):
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
            LayerGridWarp(s['scale'], s['bias'], 4) for s in act_stats
        ])
        self.l3_residual = nn.Parameter(torch.zeros(32, 32))
    def forward(self, x):
        h = x; si = 0; li = 0
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                if li == 3:
                    w1 = fq_weight(mod.weight, 4)
                    residual = mod.weight - w1.detach() + self.l3_residual
                    w2 = fq_weight(residual, 4)
                    wq = w1 + w2
                else:
                    wq = fq_weight(mod.weight, 4)
                h = nn.functional.linear(h, wq, mod.bias)
                li += 1
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h


class Model_PR(nn.Module):
    """Per-channel GridWarp + Residual Quant L3 (combined best)."""
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
        self.l3_residual = nn.Parameter(torch.zeros(32, 32))

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
        h = x; si = 0; li = 0
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                if li == 3:
                    w1 = fq_weight(mod.weight, 4)
                    residual = mod.weight - w1.detach() + self.l3_residual
                    w2 = fq_weight(residual, 4)
                    wq = w1 + w2
                else:
                    wq = fq_weight(mod.weight, 4)
                h = nn.functional.linear(h, wq, mod.bias)
                li += 1
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                if si < len(self.act_warps): h = self.act_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h


# ─── Training ─────────────────────────────────────────────────────

def train_qat(model, fnet, trainloader, epochs=80, lr=3e-4,
              warp_lr=1e-3, pinn_alpha=0.7, pinn_beta=0.2):
    warp_params, weight_params = [], []
    for name, p in model.named_parameters():
        if 'warp' in name:
            warp_params.append(p)
        else:
            weight_params.append(p)
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

# ─── Helpers ───────────────────────────────────────────────────────

def collect_act_stats(model, x_calib):
    stats = []
    h = x_calib.detach()
    for mod in list(model.encoder.net) + list(model.predictor.net):
        if isinstance(mod, nn.Linear):
            with torch.no_grad(): h = mod(h)
        elif isinstance(mod, Sin):
            with torch.no_grad(): h = torch.sin(h)
            s = torch.quantile(h.abs().flatten(), 0.995).item()
            b = h.mean().item()
            stats.append({'scale': max(s, 0.05), 'bias': b})
        elif isinstance(mod, nn.Dropout): pass
    return stats

def eval_model(model, testloader):
    model.eval(); tl, pl = [], []
    with torch.no_grad():
        for x,_,y,_ in testloader:
            u = model(x); tl.append(y.numpy()); pl.append(u.numpy())
    return eval_metrix(np.concatenate(pl), np.concatenate(tl))[1]*100

def eval_fp32(model, testloader):
    model.eval(); tl, pl = [], []
    with torch.no_grad():
        for x,_,y,_ in testloader:
            u = model(x); tl.append(y.numpy()); pl.append(u.numpy())
    return eval_metrix(np.concatenate(pl), np.concatenate(tl))[1]*100

# ─── Data loaders ──────────────────────────────────────────────────

def load_xjtu(bn):
    args=argparse.Namespace(data='XJTU',batch=bn,batch_size=256,normalization_method='min-max',log_dir=None,save_folder=None)
    root='data/XJTU'; data=XJTUdata(root=root,args=args); tl,el=[],[]
    for f in os.listdir(root):
        if bn in f:
            p=os.path.join(root,f)
            if '4' in f or '8' in f: el.append(p)
            else: tl.append(p)
    tr=data.read_all(specific_path_list=tl); te=data.read_all(specific_path_list=el)
    return {'train':tr['train_2'],'test':te['test_3']}

def load_tju(bi):
    args=argparse.Namespace(data='TJU',in_same_batch=True,batch=bi,batch_size=512,normalization_method='min-max',log_dir=None,save_folder=None)
    root='data/TJU'; data=TJUdata(root=root,args=args)
    mm=[(5,9),(4,8),(5,9)]; br=os.path.join(root,sorted(os.listdir(root))[bi])
    files=sorted(os.listdir(br)); tl,el=[],[]
    for i,f in enumerate(files):
        fid=i+1
        if fid%10==mm[bi][0] or fid%10==mm[bi][1]: el.append(os.path.join(br,f))
        else: tl.append(os.path.join(br,f))
    tr=data.read_all(specific_path_list=tl); te=data.read_all(specific_path_list=el)
    return {'train':tr['train_2'],'test':te['test_3']}

def load_mit():
    args=argparse.Namespace(data='MIT',batch_size=512,normalization_method='min-max',log_dir=None,save_folder=None)
    root='data/MIT'; tl,el=[],[]
    for b in ['2017-05-12','2017-06-30','2018-04-12']:
        br=os.path.join(root,b)
        for f in os.listdir(br):
            fid=int(f.split('-')[-1].split('.')[0])
            if fid%5==0: el.append(os.path.join(br,f))
            else: tl.append(os.path.join(br,f))
    data=MITdata(root=root,args=args)
    tr=data.read_all(specific_path_list=tl); te=data.read_all(specific_path_list=el)
    return {'train':tr['train_2'],'test':te['test_3']}

# ─── Run one model ─────────────────────────────────────────────────

def run_one(model_path, dataloader, pinn_alpha, pinn_beta):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    teacher = Solution_u(); teacher.load_state_dict(ckpt['solution_u']); teacher.eval()
    fnet = MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60, droupout=0.2)
    fnet.load_state_dict(ckpt['dynamical_F']); fnet.eval()

    fp32_mape = eval_fp32(teacher, dataloader['test'])
    all_x = torch.cat([x for x,_,_,_ in dataloader['train']])
    act_stats = collect_act_stats(teacher, all_x[:256])

    results = {'fp32': fp32_mape}

    # B: Per-layer GridWarp (old baseline)
    m = Model_B(act_stats); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['B'] = eval_model(m, dataloader['test'])

    # P: Per-channel GridWarp
    m = Model_P(); m.load_state_dict(ckpt['solution_u'], strict=False)
    m.init_warps(teacher, all_x[:256])
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['P'] = eval_model(m, dataloader['test'])

    # R: Per-layer GridWarp + Residual Quant L3
    m = Model_R(act_stats); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['R'] = eval_model(m, dataloader['test'])

    # PR: Per-channel GridWarp + Residual Quant L3 (combined)
    m = Model_PR(); m.load_state_dict(ckpt['solution_u'], strict=False)
    m.init_warps(teacher, all_x[:256])
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['PR'] = eval_model(m, dataloader['test'])

    return results

# ─── Main ──────────────────────────────────────────────────────────

def main():
    tasks = [
        ('XJTU_2C', 'models/checkpoints/XJTU/0-0', lambda: load_xjtu('2C'), 0.7, 0.2),
        ('TJU_NCM', 'models/checkpoints/TJU/1-1', lambda: load_tju(1), 1.0, 0.05),
        ('MIT', 'models/checkpoints/MIT', load_mit, 1.0, 0.02),
    ]

    all_results = {}
    for ds, mdir, load_fn, alpha, beta in tasks:
        print(f"\n{'='*70}\n  {ds}\n{'='*70}", flush=True)
        dl = load_fn(); runs = []
        for e in range(1, 11):
            mp = os.path.join(mdir, f'Experiment{e}', 'model.pth')
            if not os.path.exists(mp): continue
            print(f"  run {e}/10 ...", end=' ', flush=True)
            t0 = time.time()
            r = run_one(mp, dl, alpha, beta)
            runs.append(r)
            print(f"FP32={r['fp32']:.2f}  B={r['B']:.2f}  P={r['P']:.2f}  "
                  f"R={r['R']:.2f}  PR={r['PR']:.2f}  ({time.time()-t0:.0f}s)")

        if runs:
            means = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
            stds = {k: np.std([r[k] for r in runs]) for k in runs[0]}
            all_results[ds] = {'mean': means, 'std': stds, 'n': len(runs)}
            print(f"  MEAN: B={means['B']:.3f}  P={means['P']:.3f}  "
                  f"R={means['R']:.3f}  PR={means['PR']:.3f}")

        os.makedirs('results', exist_ok=True)
        with open('results/w4a4_best.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'='*80}")
    print("B=LayerGW  P=ChannelGW  R=LayerGW+ResQ  PR=ChannelGW+ResQ")
    print(f"{'='*80}")
    for ds, v in all_results.items():
        m = v['mean']; s = v['std']
        print(f"{ds:<12s} FP32={m['fp32']:.2f}  B={m['B']:.2f}±{s['B']:.2f}  "
              f"P={m['P']:.2f}±{s['P']:.2f}  R={m['R']:.2f}±{s['R']:.2f}  "
              f"PR={m['PR']:.2f}±{s['PR']:.2f}")


if __name__ == '__main__':
    main()
