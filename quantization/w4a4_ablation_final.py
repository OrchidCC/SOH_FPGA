#!/usr/bin/env python3
"""Final W4A4 Ablation Study + Visualization.

Ablation chain (each step adds one component):
  PTQ:  Post-training quantization (no training, just round)
  QAT:  Quantization-aware training with STE + PINN loss
  +LGW: QAT + per-layer GridWarp (scalar scale/bias per sin layer)
  +CGW: QAT + per-channel GridWarp (per-neuron scale/bias)

Datasets:
  XJTU (6 splits), TJU (3 splits), MIT, HUST

By default the script resumes from an existing results JSON and only runs
missing datasets. Use --force to recompute selected datasets.

Visualizations:
  1. Bar chart: ablation across all datasets
  2. Per-channel activation distribution before/after GridWarp
  3. Quantization error heatmap for L3 (most sensitive layer)
"""

import sys, os, copy, time, json, math, numpy as np, torch, torch.nn as nn
from torch.autograd import grad, Function
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.Model import Solution_u, MLP, Sin
from utils.util import eval_metrix
from utils.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def fq_act(x, ab):
    qm = 2**(ab-1)-1
    with torch.no_grad():
        s = x.abs().amax().clamp(min=1e-12) / qm
    return FakeQuantSTE.apply(x, s, -qm, qm)

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

# ─── 4 Models for ablation ────────────────────────────────────────

class PTQModel(nn.Module):
    """PTQ: just round weights and activations, no training."""
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
    def forward(self, x):
        h = x
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                h = nn.functional.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                h = fq_act(h, 4)
            elif isinstance(mod, nn.Dropout): pass
        return h


class QATModel(nn.Module):
    """QAT: fake quant with STE, uniform activation quant."""
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
    def forward(self, x):
        h = x
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                h = nn.functional.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                h = fq_act(h, 4)
            elif isinstance(mod, nn.Dropout): pass
        return h


class LGWModel(nn.Module):
    """QAT + per-layer GridWarp."""
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


class CGWModel(nn.Module):
    """QAT + per-channel GridWarp."""
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
        h = x_calib.detach().to(next(self.parameters()).device); si = 0
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

# ─── Training ─────────────────────────────────────────────────────

def train_qat(model, fnet, trainloader, epochs=80, lr=3e-4,
              warp_lr=1e-3, pinn_alpha=0.7, pinn_beta=0.2):
    model = model.to(device)
    fnet = fnet.to(device)
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
            x1 = x1.to(device).requires_grad_(True)
            x2 = x2.to(device).requires_grad_(True)
            y1 = y1.to(device)
            y2 = y2.to(device)
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
    model = model.to(device)
    h = x_calib.detach().to(device)
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

def collect_per_channel_acts(model, x_calib):
    """Collect per-channel sin activation values for visualization."""
    activations = []
    model = model.to(device)
    h = x_calib.detach().to(device)
    for mod in list(model.encoder.net) + list(model.predictor.net):
        if isinstance(mod, nn.Linear):
            with torch.no_grad(): h = mod(h)
        elif isinstance(mod, Sin):
            with torch.no_grad(): h = torch.sin(h)
            activations.append(h.detach().cpu().clone())
        elif isinstance(mod, nn.Dropout): pass
    return activations

def eval_model(model, testloader):
    model = model.to(device)
    model.eval(); tl, pl = [], []
    with torch.no_grad():
        for x,_,y,_ in testloader:
            u = model(x.to(device))
            tl.append(y.numpy())
            pl.append(u.detach().cpu().numpy())
    return eval_metrix(np.concatenate(pl), np.concatenate(tl))[1]*100

def eval_fp32(model, testloader):
    model = model.to(device)
    model.eval(); tl, pl = [], []
    with torch.no_grad():
        for x,_,y,_ in testloader:
            u = model(x.to(device))
            tl.append(y.numpy())
            pl.append(u.detach().cpu().numpy())
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

def load_hust():
    args=argparse.Namespace(data='HUST',batch_size=512,normalization_method='min-max',log_dir=None,save_folder=None)
    root = 'data/HUST'
    test_id = ['1-4', '1-8', '2-4', '2-8',
               '3-4', '3-8', '4-4', '4-8',
               '5-4', '5-7', '6-4', '6-8',
               '7-4', '7-8', '8-4', '8-8',
               '9-4', '9-8', '10-4', '10-8']
    data = HUSTdata(root=root, args=args)
    tl, el = [], []
    for f in os.listdir(root):
        p = os.path.join(root, f)
        if f[:-4] in test_id:
            el.append(p)
        else:
            tl.append(p)
    tr = data.read_all(specific_path_list=tl)
    te = data.read_all(specific_path_list=el)
    return {'train': tr['train_2'], 'test': te['test_3']}

# ─── Visualization ─────────────────────────────────────────────────

def plot_ablation_bar(all_results, save_path='results/fig_ablation_bar.pdf'):
    """Bar chart: MAPE for each method across datasets."""
    datasets = list(all_results.keys())
    methods = ['FP32', 'PTQ', 'QAT', 'LGW', 'CGW']
    labels = ['FP32', 'PTQ', 'QAT', '+Layer GW', '+Channel GW']
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

    n_ds = len(datasets)
    n_m = len(methods)
    x = np.arange(n_ds)
    width = 0.15

    fig_width = max(14, 1.2 * n_ds)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        key = 'fp32' if method == 'FP32' else method
        vals = [all_results[ds]['mean'][key] for ds in datasets]
        errs = [all_results[ds]['std'].get(key, 0) for ds in datasets]
        ax.bar(x + (i - n_m/2 + 0.5)*width, vals, width, yerr=errs,
               label=label, color=color, capsize=2, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.replace('XJTU_','').replace('TJU_','TJU-') for ds in datasets],
                        rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=10, ncol=5, loc='upper left')
    ax.set_ylim(0, max(ax.get_ylim()[1], 5))
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('W4A4 Ablation: Quantization Method Comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_channel_distribution(fp_model, x_calib, save_path='results/fig_channel_dist.pdf'):
    """Show per-channel sin activation distributions to motivate per-channel GridWarp."""
    acts = collect_per_channel_acts(fp_model, x_calib)
    # Plot sin3 (before L3, the most sensitive layer) — 32 channels
    sin3 = acts[2].numpy()  # (N, 32)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Top: box plot of per-channel distributions
    ax = axes[0]
    bp = ax.boxplot([sin3[:, i] for i in range(32)], patch_artist=True,
                     showfliers=False, medianprops=dict(color='red', linewidth=1.5))
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)
    ax.set_xlabel('Channel index', fontsize=11)
    ax.set_ylabel('sin(z) value', fontsize=11)
    ax.set_title('Sin3 output: per-channel distribution (before L3)', fontsize=12)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Bottom: show uniform A4 grid vs actual distributions for 4 example channels
    ax = axes[1]
    qm = 7
    example_chs = [0, 8, 16, 24]
    colors_ch = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for idx, (ch, c) in enumerate(zip(example_chs, colors_ch)):
        vals = sin3[:, ch]
        ax.hist(vals, bins=30, alpha=0.5, color=c, label=f'Ch {ch}', density=True)
    # Show uniform A4 grid (assuming absmax of full layer)
    layer_max = np.abs(sin3).max()
    grid = np.linspace(-layer_max, layer_max, 15)
    for g in grid:
        ax.axvline(x=g, color='gray', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Activation value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Per-layer A4 grid (gray lines) vs per-channel distributions', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_quant_error_improvement(all_results, save_path='results/fig_improvement.pdf'):
    """Show improvement from per-layer to per-channel GridWarp vs dataset size."""
    datasets = list(all_results.keys())
    sizes = [all_results[ds].get('train_size', 0) for ds in datasets]
    lgw = [all_results[ds]['mean']['LGW'] for ds in datasets]
    cgw = [all_results[ds]['mean']['CGW'] for ds in datasets]
    improvement = [l - c for l, c in zip(lgw, cgw)]

    fig_width = max(8, 0.75 * len(datasets))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    ax.bar(range(len(datasets)), improvement, color='#9b59b6', edgecolor='white')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([ds.replace('XJTU_','').replace('TJU_','TJU-') for ds in datasets],
                        rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('MAPE improvement (%)', fontsize=12)
    ax.set_title('Per-channel vs Per-layer GridWarp: improvement per dataset', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ─── Run one model ─────────────────────────────────────────────────

def run_one(model_path, dataloader, pinn_alpha, pinn_beta):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    teacher = Solution_u().to(device); teacher.load_state_dict(ckpt['solution_u']); teacher.eval()
    fnet = MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60, droupout=0.2).to(device)
    fnet.load_state_dict(ckpt['dynamical_F']); fnet.eval()

    fp32_mape = eval_fp32(teacher, dataloader['test'])
    all_x = torch.cat([x for x,_,_,_ in dataloader['train']])
    act_stats = collect_act_stats(teacher, all_x[:256])

    results = {'fp32': fp32_mape}

    # PTQ: no training, just quantize
    m = PTQModel().to(device); m.load_state_dict(ckpt['solution_u'], strict=False); m.eval()
    results['PTQ'] = eval_model(m, dataloader['test'])

    # QAT: train with uniform activation quant
    m = QATModel().to(device); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['QAT'] = eval_model(m, dataloader['test'])

    # +LGW: QAT + per-layer GridWarp
    m = LGWModel(act_stats).to(device); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['LGW'] = eval_model(m, dataloader['test'])

    # +CGW: QAT + per-channel GridWarp
    m = CGWModel().to(device); m.load_state_dict(ckpt['solution_u'], strict=False)
    m.init_warps(teacher, all_x[:256])
    m = train_qat(m, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['CGW'] = eval_model(m, dataloader['test'])

    return results, teacher, all_x

# ─── Main ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Run the W4A4 ablation on all datasets.')
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='Optional dataset names to run, e.g. XJTU_2C TJU_NCM MIT HUST',
    )
    parser.add_argument(
        '--results-file',
        default='results/w4a4_ablation_final.json',
        help='Path to the ablation JSON file.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Recompute datasets even if they are already present in the results JSON.',
    )
    parser.add_argument(
        '--skip-plots',
        action='store_true',
        help='Skip visualization generation after saving the JSON.',
    )
    return parser.parse_args()


def build_tasks():
    tasks = []
    for i, bn in enumerate(['2C','3C','R2.5','R3','RW','satellite']):
        tasks.append((f'XJTU_{bn}', f'models/checkpoints/XJTU/{i}-{i}', lambda bn=bn: load_xjtu(bn), 0.7, 0.2))
    for bi, nm in enumerate(['NCA','NCM','NCM_NCA']):
        tasks.append((f'TJU_{nm}', f'models/checkpoints/TJU/{bi}-{bi}', lambda bi=bi: load_tju(bi), 1.0, 0.05))
    tasks.append(('MIT', 'models/checkpoints/MIT', load_mit, 1.0, 0.02))
    tasks.append(('HUST', 'models/checkpoints/HUST', load_hust, 0.5, 0.2))
    return tasks


def load_existing_results(results_file):
    if not os.path.exists(results_file):
        return {}
    with open(results_file) as f:
        return json.load(f)


def save_results(results_file, all_results):
    os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)


def main():
    args = parse_args()
    print(f"Using device: {device}")
    tasks = build_tasks()
    if args.datasets:
        wanted = set(args.datasets)
        tasks = [task for task in tasks if task[0] in wanted]
        missing = wanted - {task[0] for task in tasks}
        if missing:
            raise ValueError(f'Unknown datasets requested: {sorted(missing)}')

    all_results = load_existing_results(args.results_file)
    first_teacher = None
    first_x = None

    for ds, mdir, load_fn, alpha, beta in tasks:
        if ds in all_results and not args.force:
            print(f"\n{'='*70}\n  {ds}\n{'='*70}")
            print("  [skip] already present in results JSON")
            continue
        print(f"\n{'='*70}\n  {ds}\n{'='*70}", flush=True)
        dl = load_fn(); runs = []
        train_size = sum(x.shape[0] for x,_,_,_ in dl['train'])

        for e in range(1, 11):
            mp = os.path.join(mdir, f'Experiment{e}', 'model.pth')
            if not os.path.exists(mp): continue
            print(f"  run {e}/10 ...", end=' ', flush=True)
            t0 = time.time()
            r, teacher, x_all = run_one(mp, dl, alpha, beta)
            runs.append(r)
            # Save first model for visualization
            if first_teacher is None:
                first_teacher = teacher
                first_x = x_all[:256]
            print(f"FP32={r['fp32']:.2f}  PTQ={r['PTQ']:.2f}  QAT={r['QAT']:.2f}  "
                  f"LGW={r['LGW']:.2f}  CGW={r['CGW']:.2f}  ({time.time()-t0:.0f}s)")

        if runs:
            means = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
            stds = {k: np.std([r[k] for r in runs]) for k in runs[0]}
            all_results[ds] = {'mean': means, 'std': stds, 'n': len(runs), 'train_size': train_size}
            print(f"  MEAN: PTQ={means['PTQ']:.2f}  QAT={means['QAT']:.2f}  "
                  f"LGW={means['LGW']:.2f}  CGW={means['CGW']:.2f}")

        save_results(args.results_file, all_results)

    # ── Summary table ──
    print(f"\n{'='*90}")
    print("W4A4 Final Ablation (MAPE %)")
    print(f"{'='*90}")
    print(f"{'Dataset':<16s} {'FP32':>6s} {'PTQ':>8s} {'QAT':>8s} {'+LGW':>8s} {'+CGW':>8s}")
    print("-" * 55)
    for ds, v in all_results.items():
        m = v['mean']
        print(f"{ds:<16s} {m['fp32']:>6.2f} {m['PTQ']:>8.2f} {m['QAT']:>8.2f} "
              f"{m['LGW']:>8.2f} {m['CGW']:>8.2f}")
    keys = ['fp32', 'PTQ', 'QAT', 'LGW', 'CGW']
    avgs = {k: np.mean([v['mean'][k] for v in all_results.values()]) for k in keys}
    print("-" * 55)
    print(f"{'Average':<16s} {avgs['fp32']:>6.2f} {avgs['PTQ']:>8.2f} {avgs['QAT']:>8.2f} "
          f"{avgs['LGW']:>8.2f} {avgs['CGW']:>8.2f}")

    # ── Visualizations ──
    if args.skip_plots:
        return
    print("\nGenerating visualizations...")
    plot_ablation_bar(all_results)
    if first_teacher is not None and first_x is not None:
        plot_channel_distribution(first_teacher, first_x)
    plot_quant_error_improvement(all_results)
    print("Done.")


if __name__ == '__main__':
    main()
