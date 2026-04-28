"""
W4A4 Ablation Study: Full datasets (XJTU 6 batches + TJU 3 batches + MIT).

Ablation:
  A: QAT only (no GridWarp, no distillation)
  B: QAT + GridWarp (no distillation)
  C: QAT + derivative distillation (no GridWarp)
  D: QAT + GridWarp + derivative distillation (full method)
"""

import sys, os, copy, time, json, math, numpy as np, torch, torch.nn as nn
from torch.autograd import grad, Function
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.Model import Solution_u, MLP, Sin
from utils.util import eval_metrix
from utils.dataloader import XJTUdata, TJUdata, MITdata
import argparse

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

def fq_act(x, ab):
    qm = 2**(ab-1)-1
    with torch.no_grad():
        s = x.abs().amax().clamp(min=1e-12) / qm
    return FakeQuantSTE.apply(x, s, -qm, qm)

# ─── GridWarp ──────────────────────────────────────────────────────

class GridWarp(nn.Module):
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

# ─── Models ────────────────────────────────────────────────────────

class UniformModel(nn.Module):
    """W4A4 with uniform activation quantization."""
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
                h = torch.sin(h); h = fq_act(h, 4)
            elif isinstance(mod, nn.Dropout): pass
        return h

class GridWarpModel(nn.Module):
    """W4A4 with GridWarp activation quantization."""
    def __init__(self, act_stats=None):
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
        if act_stats is None:
            act_stats = [{'scale': 0.3, 'bias': 0.0}] * 3
        self.grid_warps = nn.ModuleList([
            GridWarp(s['scale'], s['bias'], 4) for s in act_stats
        ])
    def forward(self, x):
        h = x; si = 0
        for mod in list(self.encoder.net) + list(self.predictor.net):
            if isinstance(mod, nn.Linear):
                h = nn.functional.linear(h, fq_weight(mod.weight, 4), mod.bias)
            elif isinstance(mod, Sin):
                h = torch.sin(h)
                if si < len(self.grid_warps): h = self.grid_warps[si](h)
                si += 1
            elif isinstance(mod, nn.Dropout): pass
        return h

# ─── Training ─────────────────────────────────────────────────────

def train_qat(model, teacher, fnet, trainloader, epochs=80, lr=3e-4,
              warp_lr=1e-3, pinn_alpha=0.7, pinn_beta=0.2,
              use_deriv_distill=False, distill_alpha=0.5):

    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)

    # Optimizer
    warp_params, other_params = [], []
    for p in model.parameters():
        if any(p is gp for gw in getattr(model, 'grid_warps', []) for gp in gw.parameters()):
            warp_params.append(p)
        else:
            other_params.append(p)
    if warp_params:
        opt = torch.optim.Adam([{'params': other_params, 'lr': lr},
                                 {'params': warp_params, 'lr': warp_lr}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

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

            # Student derivatives
            u1t_s = grad(u1_s.sum(), x1[:,-1:], create_graph=True, allow_unused=True)[0]
            u1x_s = grad(u1_s.sum(), x1[:,:-1], create_graph=True, allow_unused=True)[0]
            if u1t_s is None: u1t_s = torch.zeros_like(u1_s)
            if u1x_s is None: u1x_s = torch.zeros_like(x1[:,:-1])
            u2t_s = grad(u2_s.sum(), x2[:,-1:], create_graph=True, allow_unused=True)[0]
            u2x_s = grad(u2_s.sum(), x2[:,:-1], create_graph=True, allow_unused=True)[0]
            if u2t_s is None: u2t_s = torch.zeros_like(u2_s)
            if u2x_s is None: u2x_s = torch.zeros_like(x2[:,:-1])

            # PINN loss
            loss_data = 0.5*lf(u1_s, y1) + 0.5*lf(u2_s, y2)
            with torch.no_grad():
                F1 = fnet(torch.cat([x1.detach(), u1_s.detach(), u1x_s.detach(), u1t_s.detach()], 1))
                F2 = fnet(torch.cat([x2.detach(), u2_s.detach(), u2x_s.detach(), u2t_s.detach()], 1))
            loss_pde = 0.5*lf(u1t_s-F1, torch.zeros_like(F1)) + 0.5*lf(u2t_s-F2, torch.zeros_like(F2))
            loss_phys = relu(torch.mul(u2_s-u1_s, y1-y2)).sum()
            loss = loss_data + pinn_alpha*loss_pde + pinn_beta*loss_phys

            # Derivative distillation
            if use_deriv_distill and teacher is not None:
                x1_t = x1.detach().requires_grad_(True)
                x2_t = x2.detach().requires_grad_(True)
                u1_t = teacher(x1_t); u2_t = teacher(x2_t)
                u1t_t = grad(u1_t.sum(), x1_t[:,-1:], create_graph=False, allow_unused=True)[0]
                u1x_t = grad(u1_t.sum(), x1_t[:,:-1], create_graph=False, allow_unused=True)[0]
                u2t_t = grad(u2_t.sum(), x2_t[:,-1:], create_graph=False, allow_unused=True)[0]
                u2x_t = grad(u2_t.sum(), x2_t[:,:-1], create_graph=False, allow_unused=True)[0]
                if u1t_t is None: u1t_t = torch.zeros_like(u1t_s)
                if u1x_t is None: u1x_t = torch.zeros_like(u1x_s)
                if u2t_t is None: u2t_t = torch.zeros_like(u2t_s)
                if u2x_t is None: u2x_t = torch.zeros_like(u2x_s)
                loss_dd = (0.5*lf(u1t_s, u1t_t.detach()) + 0.5*lf(u2t_s, u2t_t.detach())
                          + 0.5*lf(u1x_s, u1x_t.detach()) + 0.5*lf(u2x_s, u2x_t.detach()))
                loss = loss + distill_alpha * loss_dd

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
    fnet = MLP(input_dim=35,output_dim=1,layers_num=3,hidden_dim=60,droupout=0.2)
    fnet.load_state_dict(ckpt['dynamical_F']); fnet.eval()

    fp32_mape = eval_fp32(teacher, dataloader['test'])
    all_x = [x for x,_,_,_ in dataloader['train']]
    act_stats = collect_act_stats(teacher, torch.cat(all_x)[:256])

    results = {'fp32': fp32_mape}

    # A: QAT only
    m = UniformModel(); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, None, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['A'] = eval_model(m, dataloader['test'])

    # B: QAT + GridWarp
    m = GridWarpModel(act_stats); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, None, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta)
    results['B'] = eval_model(m, dataloader['test'])

    # C: QAT + derivative distillation
    m = UniformModel(); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, teacher, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta,
                   use_deriv_distill=True)
    results['C'] = eval_model(m, dataloader['test'])

    # D: QAT + GridWarp + derivative distillation (full)
    m = GridWarpModel(act_stats); m.load_state_dict(ckpt['solution_u'], strict=False)
    m = train_qat(m, teacher, fnet, dataloader['train'], pinn_alpha=pinn_alpha, pinn_beta=pinn_beta,
                   use_deriv_distill=True)
    results['D'] = eval_model(m, dataloader['test'])

    return results

# ─── Main ──────────────────────────────────────────────────────────

def main():
    tasks = []
    for i, bn in enumerate(['2C','3C','R2.5','R3','RW','satellite']):
        tasks.append((f'XJTU_{bn}', f'models/checkpoints/XJTU/{i}-{i}', lambda bn=bn: load_xjtu(bn), 0.7, 0.2))
    for bi, nm in enumerate(['NCA','NCM','NCM_NCA']):
        tasks.append((f'TJU_{nm}', f'models/checkpoints/TJU/{bi}-{bi}', lambda bi=bi: load_tju(bi), 1.0, 0.05))
    tasks.append(('MIT', 'models/checkpoints/MIT', load_mit, 1.0, 0.02))

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
            print(f"FP32={r['fp32']:.2f}  A={r['A']:.2f}  B={r['B']:.2f}  "
                  f"C={r['C']:.2f}  D={r['D']:.2f}  ({time.time()-t0:.0f}s)")

        if runs:
            means = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
            stds = {k: np.std([r[k] for r in runs]) for k in runs[0]}
            all_results[ds] = {'mean': means, 'std': stds, 'n': len(runs)}
            print(f"  MEAN: A={means['A']:.3f}±{stds['A']:.3f}  B={means['B']:.3f}±{stds['B']:.3f}  "
                  f"C={means['C']:.3f}±{stds['C']:.3f}  D={means['D']:.3f}±{stds['D']:.3f}")

        os.makedirs('results', exist_ok=True)
        with open('results/w4a4_ablation.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)

    # Summary table
    print(f"\n{'='*85}")
    print("W4A4 Ablation (MAPE %): A=QAT, B=+GridWarp, C=+DerivDistill, D=+Both")
    print(f"{'='*85}")
    print(f"{'Dataset':<14s} {'FP32':>6s} {'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s}")
    print("-" * 50)
    for ds, v in all_results.items():
        m = v['mean']; s = v['std']
        print(f"{ds:<14s} {m['fp32']:>6.2f} {m['A']:>5.2f}±{s['A']:.2f} {m['B']:>5.2f}±{s['B']:.2f} "
              f"{m['C']:>5.2f}±{s['C']:.2f} {m['D']:>5.2f}±{s['D']:.2f}")

    # Average across all datasets
    all_A = [v['mean']['A'] for v in all_results.values()]
    all_B = [v['mean']['B'] for v in all_results.values()]
    all_C = [v['mean']['C'] for v in all_results.values()]
    all_D = [v['mean']['D'] for v in all_results.values()]
    all_fp = [v['mean']['fp32'] for v in all_results.values()]
    print("-" * 50)
    print(f"{'Average':<14s} {np.mean(all_fp):>6.2f} {np.mean(all_A):>8.2f} {np.mean(all_B):>8.2f} "
          f"{np.mean(all_C):>8.2f} {np.mean(all_D):>8.2f}")

    print(f"\nSaved to results/w4a4_ablation.json")


if __name__ == '__main__':
    main()
