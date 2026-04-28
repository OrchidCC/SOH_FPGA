"""
PCR vs PTQ on ALL datasets (XJTU, TJU, MIT, HUST) × all FPGA configs.
Configs: W8A8, W6A6, W5A8, W4A8, W4A6
"""

import sys
import os
import copy
import time
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.Model import Solution_u, MLP, Sin
from utils.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
from utils.util import eval_metrix

device = 'cpu'

CONFIGS = [
    ('W8A8', 8, 8),
    ('W6A6', 6, 6),
    ('W5A8', 5, 8),
    ('W4A8', 4, 8),
    ('W4A6', 4, 6),
]


# ─── Data loaders ──────────────────────────────────────────────────

def load_xjtu_data(batch_name):
    args = argparse.Namespace(data='XJTU', batch=batch_name, batch_size=256,
                              normalization_method='min-max', log_dir=None, save_folder=None)
    root = 'data/XJTU'
    data = XJTUdata(root=root, args=args)
    train_list, test_list = [], []
    for f in os.listdir(root):
        if batch_name in f:
            path = os.path.join(root, f)
            if '4' in f or '8' in f:
                test_list.append(path)
            else:
                train_list.append(path)
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {'train': train_loader['train_2'], 'valid': train_loader['valid_2'],
            'test': test_loader['test_3']}


def load_tju_data(batch_idx):
    args = argparse.Namespace(data='TJU', in_same_batch=True, batch=batch_idx,
                              batch_size=512, normalization_method='min-max',
                              log_dir=None, save_folder=None)
    root = 'data/TJU'
    data = TJUdata(root=root, args=args)
    mod = [(5, 9), (4, 8), (5, 9)]
    batchs = sorted(os.listdir(root))
    batch_root = os.path.join(root, batchs[batch_idx])
    files = sorted(os.listdir(batch_root))
    train_list, test_list = [], []
    for i, f in enumerate(files):
        fid = i + 1
        if fid % 10 == mod[batch_idx][0] or fid % 10 == mod[batch_idx][1]:
            test_list.append(os.path.join(batch_root, f))
        else:
            train_list.append(os.path.join(batch_root, f))
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {'train': train_loader['train_2'], 'valid': train_loader['valid_2'],
            'test': test_loader['test_3']}


def load_mit_data():
    args = argparse.Namespace(data='MIT', batch_size=512,
                              normalization_method='min-max',
                              log_dir=None, save_folder=None)
    root = 'data/MIT'
    train_list, test_list = [], []
    for batch in ['2017-05-12', '2017-06-30', '2018-04-12']:
        batch_root = os.path.join(root, batch)
        for f in os.listdir(batch_root):
            fid = int(f.split('-')[-1].split('.')[0])
            if fid % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    data = MITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    return {'train': trainloader['train_2'], 'valid': trainloader['valid_2'],
            'test': testloader['test_3']}


def load_hust_data():
    args = argparse.Namespace(data='HUST', batch_size=512,
                              normalization_method='min-max',
                              log_dir=None, save_folder=None)
    test_id = ['1-4', '1-8', '2-4', '2-8', '3-4', '3-8', '4-4', '4-8',
               '5-4', '5-7', '6-4', '6-8', '7-4', '7-8', '8-4', '8-8',
               '9-4', '9-8', '10-4', '10-8']
    data = HUSTdata(root='data/HUST', args=args)
    train_list, test_list = [], []
    for f in os.listdir('data/HUST'):
        if f[:-4] in test_id:
            test_list.append(f'data/HUST/{f}')
        else:
            train_list.append(f'data/HUST/{f}')
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    return {'train': trainloader['train_2'], 'valid': trainloader['valid_2'],
            'test': testloader['test_3']}


# ─── Quantization (parameterized bit-width) ────────────────────────

def per_tensor_quant_candidates(tensor, w_bits):
    qmax = 2 ** (w_bits - 1) - 1
    abs_max = tensor.abs().max()
    if abs_max.item() <= 0.0:
        zeros = torch.zeros_like(tensor)
        return {'down': tensor.clone(), 'up': tensor.clone(),
                'ptq': tensor.clone(), 'mid': tensor.clone(), 'half': zeros}
    scale = abs_max / qmax
    scaled = tensor / scale
    down = torch.floor(scaled) * scale
    up = torch.ceil(scaled) * scale
    ptq = torch.round(scaled) * scale
    mid = 0.5 * (down + up)
    half = 0.5 * (up - down)
    return {'down': down, 'up': up, 'ptq': ptq, 'mid': mid, 'half': half}


def build_quant_layout(model, w_bits):
    parts = {k: [] for k in ['w', 'down', 'up', 'ptq', 'mid', 'half']}
    meta = []
    for name, p in model.named_parameters():
        cand = per_tensor_quant_candidates(p.detach(), w_bits)
        parts['w'].append(p.detach().reshape(-1))
        for k in ['down', 'up', 'ptq', 'mid', 'half']:
            parts[k].append(cand[k].reshape(-1))
        meta.append({'name': name, 'shape': tuple(p.shape), 'numel': p.numel()})
    flat = {k: torch.cat(v) for k, v in parts.items()}
    z_ptq = torch.where(
        (flat['ptq'] - flat['up']).abs() <= (flat['ptq'] - flat['down']).abs(),
        torch.ones_like(flat['ptq']), -torch.ones_like(flat['ptq']))
    z_ptq = torch.where(flat['half'] > 0, z_ptq, torch.ones_like(z_ptq))
    return {
        'meta': meta, 'flat_w': flat['w'], 'flat_ptq': flat['ptq'],
        'flat_mid': flat['mid'], 'flat_half': flat['half'],
        'flat_bias': flat['mid'] - flat['w'], 'z_ptq': z_ptq,
        'active_idx': torch.nonzero(flat['half'] > 0, as_tuple=False).reshape(-1),
    }


def quantize_model_ptq(model, w_bits):
    qmax = 2 ** (w_bits - 1) - 1
    q = Solution_u().to(device)
    q.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        for p in q.parameters():
            scale = p.abs().max().item() / qmax
            if scale > 0:
                p.data = torch.round(p / scale) * scale
    q.eval()
    return q


def set_flat_params(model, flat):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


# ─── Forward with activation quantization ──────────────────────────

def forward_quantized(model, x, a_bits):
    qmax_a = 2 ** (a_bits - 1) - 1
    h = x
    for module in model.encoder.net:
        if isinstance(module, nn.Linear):
            h = module(h)
        elif isinstance(module, Sin):
            h = torch.sin(h)
            h = torch.round(h * qmax_a) / qmax_a
        elif isinstance(module, nn.Dropout):
            pass
    for module in model.predictor.net:
        if isinstance(module, nn.Linear):
            h = module(h)
        elif isinstance(module, Sin):
            h = torch.sin(h)
            h = torch.round(h * qmax_a) / qmax_a
        elif isinstance(module, nn.Dropout):
            pass
    return h


def evaluate_quantized(model, dataloader, a_bits):
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for x, _, y, _ in dataloader:
            x = x.to(device)
            u = forward_quantized(model, x, a_bits)
            true_labels.append(y.numpy())
            pred_labels.append(u.cpu().numpy())
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_labels, true_labels)
    return MAPE * 100


def evaluate_fp32(model, dataloader):
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for x, _, y, _ in dataloader:
            x = x.to(device)
            u = model(x)
            true_labels.append(y.numpy())
            pred_labels.append(u.cpu().numpy())
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_labels, true_labels)
    return MAPE * 100


# ─── PDE residual & PCR (from pcr_vs_ptq.py) ──────────────────────

def compute_pde_residual(solution_u, dynamical_F, xt):
    import math
    xt = xt.clone().requires_grad_(True)
    x = xt[:, :-1]
    t = xt[:, -1:]
    u = solution_u(xt)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True, only_inputs=True, allow_unused=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True, only_inputs=True, allow_unused=True)[0]
    if u_t is None:
        u_t = torch.zeros_like(u)
    if u_x is None:
        u_x = torch.zeros_like(x)
    F_input = torch.cat([xt, u, u_x, u_t], dim=1)
    F_val = dynamical_F(F_input)
    return u_t - F_val


def stacked_physics_vector(solution_u, dynamical_F, x_data, y_data, x_pde, w_data, w_pde):
    import math
    parts = []
    data_err = (solution_u(x_data) - y_data).reshape(-1)
    parts.append(math.sqrt(w_data) * data_err)
    pde_res = compute_pde_residual(solution_u, dynamical_F, x_pde).reshape(-1)
    parts.append(math.sqrt(w_pde) * pde_res)
    return torch.cat(parts, dim=0)


def build_linearized_problem(solution_u, dynamical_F, layout, x_data, y_data, x_pde, w_data, w_pde):
    params = list(solution_u.parameters())
    s_ref = stacked_physics_vector(solution_u, dynamical_F, x_data, y_data, x_pde, w_data, w_pde)
    s_ref_cpu = s_ref.detach().cpu().float()
    bias_cpu = layout['flat_bias'].detach().cpu().float()
    half_cpu = layout['flat_half'].detach().cpu().float()
    n_rows = s_ref_cpu.numel()
    n_params = bias_cpu.numel()
    a = torch.zeros(n_rows, dtype=torch.float32)
    b_mat = torch.zeros(n_rows, n_params, dtype=torch.float32)
    t0 = time.time()
    for row in range(n_rows):
        grads = torch.autograd.grad(s_ref[row], params, retain_graph=(row + 1 < n_rows), allow_unused=True)
        flat_grad_parts = []
        for g, p in zip(grads, params):
            if g is None:
                flat_grad_parts.append(torch.zeros(p.numel()))
            else:
                flat_grad_parts.append(g.reshape(-1).detach().cpu().float())
        flat_grad = torch.cat(flat_grad_parts)
        b_mat[row].copy_(flat_grad * half_cpu)
        a[row] = s_ref_cpu[row] + torch.dot(flat_grad, bias_cpu)
        if (row + 1) % 50 == 0 or row + 1 == n_rows:
            elapsed = time.time() - t0
            print(f"        Jacobian: {row+1:4d}/{n_rows} ({elapsed:.1f}s)", flush=True)
    return a, b_mat


def solve_pcr_signs(a, b_mat, z_ptq, active_idx, max_rank=8, energy_threshold=0.995, num_sweeps=4):
    u, s, vh = torch.linalg.svd(b_mat, full_matrices=False)
    energy = s * s
    if float(energy.sum().item()) <= 0.0:
        return z_ptq.clone()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    rank_from_energy = int(torch.searchsorted(cumulative, torch.tensor(energy_threshold)).item()) + 1
    rank = min(max_rank, rank_from_energy, s.numel())
    vh_k = vh[:rank].float()
    s_k = s[:rank].float()
    d_k = (u[:, :rank].T @ a).float()
    a_norm_sq = torch.dot(a, a).float()

    def objective(z):
        u_v = vh_k @ z
        sigma_d = s_k * d_k
        sigma_sq = s_k * s_k
        return a_norm_sq + 2.0 * torch.dot(sigma_d, u_v) + torch.dot(sigma_sq * u_v, u_v)

    def coord_descent(z_init):
        z = z_init.clone()
        u_v = vh_k @ z
        sigma_d = s_k * d_k
        sigma_sq = s_k * s_k
        influence = torch.sum((s_k[:, None] * vh_k[:, active_idx]) ** 2, dim=0)
        ordered = active_idx[torch.argsort(influence, descending=True)]
        for _ in range(num_sweeps):
            flips = 0
            grad_u = sigma_sq * u_v + sigma_d
            for idx in ordered.tolist():
                z_i = float(z[idx].item())
                v_i = vh_k[:, idx]
                delta = -4.0 * z_i * torch.dot(v_i, grad_u) + 4.0 * torch.dot(sigma_sq * v_i, v_i)
                if float(delta.item()) < -1e-10:
                    z[idx] = -z_i
                    u_v = u_v - 2.0 * z_i * v_i
                    grad_u = sigma_sq * u_v + sigma_d
                    flips += 1
            if flips == 0:
                break
        return z

    linear_scores = vh_k.T @ (s_k * d_k)
    z_linear = torch.where(linear_scores > 0, -torch.ones_like(z_ptq), torch.ones_like(z_ptq))
    z_linear = torch.where(linear_scores.abs() > 1e-12, z_linear, z_ptq)

    best_z, best_obj = None, None
    for z_init in [z_ptq.float(), z_linear.float()]:
        z_c = coord_descent(z_init)
        obj = objective(z_c)
        if best_obj is None or float(obj.item()) < best_obj:
            best_z, best_obj = z_c, float(obj.item())
    return best_z


def sample_calibration_data(dataloader, n_data=48, n_pde=48, seed=42):
    torch.manual_seed(seed)
    all_x, all_y = [], []
    for x, _, y, _ in dataloader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    perm = torch.randperm(all_x.shape[0])
    x_data = all_x[perm[:n_data]].to(device)
    y_data = all_y[perm[:n_data]].to(device)
    x_pde = all_x[perm[n_data:n_data + n_pde]].to(device)
    return x_data, y_data, x_pde


# ─── Run one model × one config ───────────────────────────────────

def run_one_config(model_path, dataloader, config_name, w_bits, a_bits):
    """Returns (ptq_mape, pcr_mape) for one model + one config."""
    # Load model
    model = Solution_u().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['solution_u'])
    model.eval()

    # Load dynamical_F for PCR
    f_net = MLP(input_dim=35, output_dim=1, layers_num=3, hidden_dim=60, droupout=0.2).to(device)
    f_net.load_state_dict(ckpt['dynamical_F'])
    f_net.eval()

    testloader = dataloader['test']
    trainloader = dataloader['train']

    # PTQ
    ptq_model = quantize_model_ptq(model, w_bits)
    ptq_mape = evaluate_quantized(ptq_model, testloader, a_bits)

    # PCR
    x_data, y_data, x_pde = sample_calibration_data(trainloader)
    layout = build_quant_layout(model, w_bits)
    a, b_mat = build_linearized_problem(model, f_net, layout, x_data, y_data, x_pde, w_data=1.0, w_pde=1.0)
    z_pcr = solve_pcr_signs(a, b_mat, layout['z_ptq'].detach().cpu().float(),
                            layout['active_idx'].detach().cpu())
    flat_pcr = layout['flat_mid'].detach().cpu().float() + layout['flat_half'].detach().cpu().float() * z_pcr

    pcr_model = Solution_u().to(device)
    pcr_model.load_state_dict(copy.deepcopy(model.state_dict()))
    set_flat_params(pcr_model, flat_pcr.to(device))
    pcr_model.eval()
    pcr_mape = evaluate_quantized(pcr_model, testloader, a_bits)

    return ptq_mape, pcr_mape


# ─── Main ──────────────────────────────────────────────────────────

def main():
    results = {}

    # Define all dataset/batch combos
    tasks = []

    # XJTU: 6 batches
    xjtu_batches = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i, bn in enumerate(xjtu_batches):
        tasks.append({
            'dataset': 'XJTU', 'batch': bn,
            'model_dir': f'models/checkpoints/XJTU/{i}-{i}',
            'load_data': lambda bn=bn: load_xjtu_data(bn),
        })

    # TJU: 3 batches
    tju_names = ['NCA', 'NCM', 'NCM_NCA']
    for bi in range(3):
        tasks.append({
            'dataset': 'TJU', 'batch': tju_names[bi],
            'model_dir': f'models/checkpoints/TJU/{bi}-{bi}',
            'load_data': lambda bi=bi: load_tju_data(bi),
        })

    # MIT: 1
    tasks.append({
        'dataset': 'MIT', 'batch': 'all',
        'model_dir': 'models/checkpoints/MIT',
        'load_data': load_mit_data,
    })

    # HUST: 1
    tasks.append({
        'dataset': 'HUST', 'batch': 'all',
        'model_dir': 'models/checkpoints/HUST',
        'load_data': load_hust_data,
    })

    for task in tasks:
        ds = task['dataset']
        bn = task['batch']
        key = f"{ds}_{bn}"
        print(f"\n{'='*60}")
        print(f"  {key}")
        print(f"{'='*60}", flush=True)

        dataloader = task['load_data']()
        results[key] = {}

        for config_name, w_bits, a_bits in CONFIGS:
            ptq_mapes, pcr_mapes = [], []

            for e in range(1, 11):
                mp = os.path.join(task['model_dir'], f'Experiment{e}', 'model.pth')
                if not os.path.exists(mp):
                    continue
                print(f"    {config_name} run {e}/10 ...", flush=True)
                ptq_m, pcr_m = run_one_config(mp, dataloader, config_name, w_bits, a_bits)
                ptq_mapes.append(ptq_m)
                pcr_mapes.append(pcr_m)
                print(f"      PTQ={ptq_m:.4f}%  PCR={pcr_m:.4f}%")

            if ptq_mapes:
                results[key][config_name] = {
                    'ptq_mean': float(np.mean(ptq_mapes)),
                    'pcr_mean': float(np.mean(pcr_mapes)),
                    'improve': float(np.mean(ptq_mapes) - np.mean(pcr_mapes)),
                    'ptq_std': float(np.std(ptq_mapes)),
                    'pcr_std': float(np.std(pcr_mapes)),
                    'n': len(ptq_mapes),
                }
                print(f"  {config_name}: PTQ={np.mean(ptq_mapes):.4f}% PCR={np.mean(pcr_mapes):.4f}% "
                      f"(improve={np.mean(ptq_mapes)-np.mean(pcr_mapes):.4f}%)")

        # Save incrementally
        os.makedirs('results', exist_ok=True)
        with open('results/all_datasets_pcr_vs_ptq.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY: PCR vs PTQ (MAPE %)")
    print(f"{'='*80}")
    header = f"{'Dataset':<16s}"
    for cn, _, _ in CONFIGS:
        header += f"  {'PTQ '+cn:>10s} {'PCR '+cn:>10s}"
    print(header)
    print("-" * 120)
    for key, cfg_results in results.items():
        line = f"{key:<16s}"
        for cn, _, _ in CONFIGS:
            if cn in cfg_results:
                r = cfg_results[cn]
                line += f"  {r['ptq_mean']:>9.2f}% {r['pcr_mean']:>9.2f}%"
            else:
                line += f"  {'N/A':>10s} {'N/A':>10s}"
        print(line)

    print(f"\nResults saved to results/all_datasets_pcr_vs_ptq.json")


if __name__ == '__main__':
    main()
