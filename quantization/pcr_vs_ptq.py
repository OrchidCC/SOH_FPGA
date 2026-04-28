"""
PCR vs PTQ 对比实验 on PINN4SOH (Nature Communications 2024)

对 Solution_u 网络做 INT8 权重量化：
1. PTQ: 标准四舍五入
2. PCR: PDE-Constrained Rounding，用 PDE 残差 (u_t - F) 作为约束目标

评估指标：MAE, MAPE, RMSE (在测试集上)
"""

import sys
import os
import copy
import time
import math
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.Model import Solution_u, MLP, Sin, PINN as PINN_SOH
from utils.dataloader import XJTUdata
from utils.util import eval_metrix

device = 'cpu'

# ─── 数据加载 ────────────────────────────────────────────────────────

def load_xjtu_data(batch_name: str):
    """加载 XJTU 数据，按 main_XJTU.py 的方式划分训练/测试"""
    args = argparse.Namespace(
        data='XJTU', batch=batch_name, batch_size=256,
        normalization_method='z-score',
        log_dir=None, save_folder=None
    )
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

    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3'],
    }


def load_pretrained(model_path: str) -> Solution_u:
    model = Solution_u().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['solution_u'])
    model.eval()
    return model


def load_dynamical_F(model_path: str, F_layers_num=3, F_hidden_dim=60) -> MLP:
    f_net = MLP(input_dim=35, output_dim=1,
                layers_num=F_layers_num, hidden_dim=F_hidden_dim,
                droupout=0.2).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    f_net.load_state_dict(ckpt['dynamical_F'])
    f_net.eval()
    return f_net


# ─── 评估 ────────────────────────────────────────────────────────────

def forward_w8a8(model: Solution_u, x: torch.Tensor, act_bits: int = 8) -> torch.Tensor:
    """W8A8 前向：权重已量化，激活也量化"""
    qmax_a = 2 ** (act_bits - 1) - 1
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


def evaluate_w8a8(model: nn.Module, dataloader, act_bits: int = 8) -> Dict[str, float]:
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for x, _, y, _ in dataloader:
            x = x.to(device)
            u = forward_w8a8(model, x, act_bits=act_bits)
            true_labels.append(y.numpy())
            pred_labels.append(u.cpu().numpy())
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_labels, true_labels)
    return {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE, 'RMSE': RMSE}


def evaluate_model(model: nn.Module, dataloader) -> Dict[str, float]:
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
    return {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE, 'RMSE': RMSE}


# ─── 量化工具 ────────────────────────────────────────────────────────

QMAX = 127


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


def per_tensor_quant_candidates(tensor: torch.Tensor):
    abs_max = tensor.abs().max()
    if abs_max.item() <= 0.0:
        zeros = torch.zeros_like(tensor)
        return {'down': tensor.clone(), 'up': tensor.clone(),
                'ptq': tensor.clone(), 'mid': tensor.clone(),
                'half': zeros}
    scale = abs_max / QMAX
    scaled = tensor / scale
    down = torch.floor(scaled) * scale
    up = torch.ceil(scaled) * scale
    ptq = torch.round(scaled) * scale
    mid = 0.5 * (down + up)
    half = 0.5 * (up - down)
    return {'down': down, 'up': up, 'ptq': ptq, 'mid': mid, 'half': half}


def build_quant_layout(model: nn.Module) -> Dict[str, object]:
    parts = {k: [] for k in ['w', 'down', 'up', 'ptq', 'mid', 'half']}
    meta = []

    for name, p in model.named_parameters():
        cand = per_tensor_quant_candidates(p.detach())
        parts['w'].append(p.detach().reshape(-1))
        for k in ['down', 'up', 'ptq', 'mid', 'half']:
            parts[k].append(cand[k].reshape(-1))
        meta.append({'name': name, 'shape': tuple(p.shape), 'numel': p.numel()})

    flat = {k: torch.cat(v) for k, v in parts.items()}

    z_ptq = torch.where(
        (flat['ptq'] - flat['up']).abs() <= (flat['ptq'] - flat['down']).abs(),
        torch.ones_like(flat['ptq']),
        -torch.ones_like(flat['ptq']),
    )
    z_ptq = torch.where(flat['half'] > 0, z_ptq, torch.ones_like(z_ptq))

    return {
        'meta': meta,
        'flat_w': flat['w'],
        'flat_ptq': flat['ptq'],
        'flat_mid': flat['mid'],
        'flat_half': flat['half'],
        'flat_bias': flat['mid'] - flat['w'],
        'z_ptq': z_ptq,
        'active_idx': torch.nonzero(flat['half'] > 0, as_tuple=False).reshape(-1),
    }


def quantize_model_ptq(model: Solution_u) -> Solution_u:
    q = Solution_u().to(device)
    q.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        for p in q.parameters():
            scale = p.abs().max().item() / QMAX
            if scale > 0:
                p.data = torch.round(p / scale) * scale
    q.eval()
    return q


# ─── PDE 残差计算 ────────────────────────────────────────────────────

def compute_pde_residual(
    solution_u: Solution_u,
    dynamical_F: MLP,
    xt: torch.Tensor,
) -> torch.Tensor:
    """计算 PDE 残差 f = u_t - F(x,t,u,u_x,u_t)"""
    xt = xt.clone().requires_grad_(True)
    x = xt[:, :-1]
    t = xt[:, -1:]

    u = solution_u(xt)

    u_t = torch.autograd.grad(u.sum(), t,
                              create_graph=True,
                              only_inputs=True,
                              allow_unused=True)[0]
    u_x = torch.autograd.grad(u.sum(), x,
                              create_graph=True,
                              only_inputs=True,
                              allow_unused=True)[0]

    if u_t is None:
        u_t = torch.zeros_like(u)
    if u_x is None:
        u_x = torch.zeros_like(x)

    F_input = torch.cat([xt, u, u_x, u_t], dim=1)
    F_val = dynamical_F(F_input)

    residual = u_t - F_val
    return residual


def compute_data_error(
    solution_u: Solution_u,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """数据拟合误差"""
    return (solution_u(x) - y).reshape(-1)


def stacked_physics_vector(
    solution_u: Solution_u,
    dynamical_F: MLP,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_pde: torch.Tensor,
    w_data: float = 1.0,
    w_pde: float = 1.0,
) -> torch.Tensor:
    """堆叠物理向量：数据误差 + PDE 残差"""
    parts = []
    # 数据误差
    data_err = compute_data_error(solution_u, x_data, y_data)
    parts.append(math.sqrt(w_data) * data_err)
    # PDE 残差
    pde_res = compute_pde_residual(solution_u, dynamical_F, x_pde).reshape(-1)
    parts.append(math.sqrt(w_pde) * pde_res)
    return torch.cat(parts, dim=0)


# ─── PCR 求解 ────────────────────────────────────────────────────────

def build_linearized_problem(
    solution_u: Solution_u,
    dynamical_F: MLP,
    layout: Dict,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_pde: torch.Tensor,
    w_data: float,
    w_pde: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """构建线性化问题 s(w_q) ≈ a + B·z"""
    params = list(solution_u.parameters())
    s_ref = stacked_physics_vector(solution_u, dynamical_F,
                                   x_data, y_data, x_pde,
                                   w_data, w_pde)
    s_ref_cpu = s_ref.detach().cpu().float()
    bias_cpu = layout['flat_bias'].detach().cpu().float()
    half_cpu = layout['flat_half'].detach().cpu().float()

    n_rows = s_ref_cpu.numel()
    n_params = bias_cpu.numel()
    a = torch.zeros(n_rows, dtype=torch.float32)
    b_mat = torch.zeros(n_rows, n_params, dtype=torch.float32)

    t0 = time.time()
    for row in range(n_rows):
        grads = torch.autograd.grad(
            s_ref[row], params,
            retain_graph=(row + 1 < n_rows),
            allow_unused=True,
        )
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
            print(f"      Jacobian rows: {row + 1:4d}/{n_rows} | elapsed={elapsed:.1f}s")

    return a, b_mat, {'n_rows': n_rows, 'n_params': n_params}


def low_rank_objective(z, a_norm_sq, vh_k, s_k, d_k):
    u = vh_k @ z
    sigma_d = s_k * d_k
    sigma_sq = s_k * s_k
    return a_norm_sq + 2.0 * torch.dot(sigma_d, u) + torch.dot(sigma_sq * u, u)


def coordinate_descent_low_rank(z_init, vh_k, s_k, d_k, a_norm_sq,
                                 active_idx, num_sweeps):
    z = z_init.clone()
    u = vh_k @ z
    sigma_d = s_k * d_k
    sigma_sq = s_k * s_k

    influence = torch.sum((s_k[:, None] * vh_k[:, active_idx]) ** 2, dim=0)
    ordered_idx = active_idx[torch.argsort(influence, descending=True)]

    total_flips = 0
    for _ in range(num_sweeps):
        flips = 0
        grad_u = sigma_sq * u + sigma_d
        for idx in ordered_idx.tolist():
            z_i = float(z[idx].item())
            v_i = vh_k[:, idx]
            delta = -4.0 * z_i * torch.dot(v_i, grad_u) + 4.0 * torch.dot(sigma_sq * v_i, v_i)
            if float(delta.item()) < -1e-10:
                z[idx] = -z_i
                u = u - 2.0 * z_i * v_i
                grad_u = sigma_sq * u + sigma_d
                flips += 1
        total_flips += flips
        if flips == 0:
            break

    obj = low_rank_objective(z, a_norm_sq, vh_k, s_k, d_k)
    return z, float(obj.item()), total_flips


def solve_pcr_signs(a, b_mat, z_ptq, active_idx, max_rank, energy_threshold,
                    num_sweeps):
    u, s, vh = torch.linalg.svd(b_mat, full_matrices=False)
    energy = s * s
    energy_total = float(energy.sum().item())

    if energy_total <= 0.0:
        return {'z': z_ptq.clone(), 'rank': 0, 'flips': 0,
                'obj_start': float((a @ a).item()),
                'obj_end': float((a @ a).item())}

    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    rank_from_energy = int(torch.searchsorted(cumulative, torch.tensor(energy_threshold)).item()) + 1
    rank = min(max_rank, rank_from_energy, s.numel())

    vh_k = vh[:rank].float()
    s_k = s[:rank].float()
    d_k = (u[:, :rank].T @ a).float()
    a_norm_sq = torch.dot(a, a).float()

    # 两个起点
    linear_scores = vh_k.T @ (s_k * d_k)
    z_linear = torch.where(linear_scores > 0, -torch.ones_like(z_ptq), torch.ones_like(z_ptq))
    z_linear = torch.where(linear_scores.abs() > 1e-12, z_linear, z_ptq)

    starts = {'ptq': z_ptq.float(), 'linear': z_linear.float()}
    best_name, best_z, best_obj, best_flips = None, None, None, 0

    for name, z_init in starts.items():
        z_c, _, flips = coordinate_descent_low_rank(
            z_init, vh_k, s_k, d_k, a_norm_sq, active_idx, num_sweeps)
        obj = low_rank_objective(z_c, a_norm_sq, vh_k, s_k, d_k)
        if best_obj is None or float(obj.item()) < best_obj:
            best_name, best_z, best_obj, best_flips = name, z_c, float(obj.item()), flips

    start_obj = low_rank_objective(z_ptq.float(), a_norm_sq, vh_k, s_k, d_k)
    return {
        'z': best_z, 'rank': rank, 'flips': best_flips,
        'start_name': best_name,
        'obj_start': float(start_obj.item()),
        'obj_end': best_obj,
    }


# ─── 采样校准点 ──────────────────────────────────────────────────────

def sample_calibration_data(dataloader, n_data=64, n_pde=64, seed=42):
    """从 dataloader 中采样校准数据"""
    torch.manual_seed(seed)
    all_x, all_y = [], []
    for x, _, y, _ in dataloader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    n = all_x.shape[0]
    perm = torch.randperm(n)

    # 数据校准点
    idx_data = perm[:n_data]
    x_data = all_x[idx_data].to(device)
    y_data = all_y[idx_data].to(device)

    # PDE 校准点（不需要标签，只需要输入）
    idx_pde = perm[n_data:n_data + n_pde]
    x_pde = all_x[idx_pde].to(device)

    return x_data, y_data, x_pde


# ─── 主实验 ──────────────────────────────────────────────────────────

def run_one(model_path: str, batch_name: str, batch_idx: int,
            args: argparse.Namespace) -> Dict:
    print(f"\n{'='*60}")
    print(f"Model: XJTU_{batch_idx}, Batch: {batch_name}")
    print(f"{'='*60}")

    # 加载模型和数据
    solution_u = load_pretrained(model_path)
    dynamical_F = load_dynamical_F(model_path)
    data = load_xjtu_data(batch_name)
    test_loader = data['test']
    train_loader = data['train']

    # FP32 baseline
    fp32_result = evaluate_model(solution_u, test_loader)
    print(f"  FP32:  MAE={fp32_result['MAE']:.6f}  MAPE={fp32_result['MAPE']*100:.4f}%  RMSE={fp32_result['RMSE']:.6f}")

    # PTQ
    ptq_model = quantize_model_ptq(solution_u)
    ptq_result = evaluate_model(ptq_model, test_loader)
    print(f"  PTQ:   MAE={ptq_result['MAE']:.6f}  MAPE={ptq_result['MAPE']*100:.4f}%  RMSE={ptq_result['RMSE']:.6f}")

    # PCR
    print(f"  PCR: 构建线性化问题...")
    x_data, y_data, x_pde = sample_calibration_data(
        train_loader, n_data=args.n_data, n_pde=args.n_pde, seed=args.seed)

    layout = build_quant_layout(solution_u)
    a, b_mat, lin_info = build_linearized_problem(
        solution_u, dynamical_F, layout,
        x_data, y_data, x_pde,
        w_data=args.w_data, w_pde=args.w_pde)

    solve_info = solve_pcr_signs(
        a=a, b_mat=b_mat,
        z_ptq=layout['z_ptq'].detach().cpu().float(),
        active_idx=layout['active_idx'].detach().cpu(),
        max_rank=args.max_rank,
        energy_threshold=args.energy_threshold,
        num_sweeps=args.coord_sweeps)

    flat_pcr = layout['flat_mid'].detach().cpu().float() + \
               layout['flat_half'].detach().cpu().float() * solve_info['z']

    pcr_model = Solution_u().to(device)
    pcr_model.load_state_dict(copy.deepcopy(solution_u.state_dict()))
    set_flat_params(pcr_model, flat_pcr.to(device))
    pcr_model.eval()

    pcr_result = evaluate_model(pcr_model, test_loader)
    print(f"  PCR W8:  MAE={pcr_result['MAE']:.6f}  MAPE={pcr_result['MAPE']*100:.4f}%  RMSE={pcr_result['RMSE']:.6f}")

    # W8A8 评估
    ptq_w8a8_result = evaluate_w8a8(ptq_model, test_loader, act_bits=8)
    print(f"  PTQ W8A8: MAE={ptq_w8a8_result['MAE']:.6f}  MAPE={ptq_w8a8_result['MAPE']*100:.4f}%  RMSE={ptq_w8a8_result['RMSE']:.6f}")

    pcr_w8a8_result = evaluate_w8a8(pcr_model, test_loader, act_bits=8)
    print(f"  PCR W8A8: MAE={pcr_w8a8_result['MAE']:.6f}  MAPE={pcr_w8a8_result['MAPE']*100:.4f}%  RMSE={pcr_w8a8_result['RMSE']:.6f}")

    print(f"\n  PCR rank={solve_info['rank']}, flips={solve_info['flips']}")

    row = {
        'batch': batch_name,
        'batch_idx': batch_idx,
        'fp32': fp32_result,
        'ptq_w8': ptq_result,
        'pcr_w8': pcr_result,
        'ptq_w8a8': ptq_w8a8_result,
        'pcr_w8a8': pcr_w8a8_result,
        'pcr_rank': solve_info['rank'],
        'pcr_flips': solve_info['flips'],
        'pcr_obj_start': solve_info['obj_start'],
        'pcr_obj_end': solve_info['obj_end'],
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', nargs='+',
                        default=['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'])
    parser.add_argument('--n-data', type=int, default=48,
                        help='数据校准点数')
    parser.add_argument('--n-pde', type=int, default=48,
                        help='PDE校准点数')
    parser.add_argument('--w-data', type=float, default=1.0)
    parser.add_argument('--w-pde', type=float, default=1.0)
    parser.add_argument('--max-rank', type=int, default=8)
    parser.add_argument('--energy-threshold', type=float, default=0.995)
    parser.add_argument('--coord-sweeps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-json', default='results/pcr_vs_ptq_soh.json')
    args = parser.parse_args()

    # batch_name -> model_path 映射
    batch_model_map = {
        '2C': ('pretrained model/model_XJTU_0.pth', 0),
        '3C': ('pretrained model/model_XJTU_1.pth', 1),
        'R2.5': ('pretrained model/model_XJTU_2.pth', 2),
        'R3': ('pretrained model/model_XJTU_3.pth', 3),
        'RW': ('pretrained model/model_XJTU_4.pth', 4),
        'satellite': ('pretrained model/model_XJTU_5.pth', 5),
    }

    rows = []
    for batch_name in args.batches:
        model_path, batch_idx = batch_model_map[batch_name]
        row = run_one(model_path, batch_name, batch_idx, args)
        rows.append(row)

    # 汇总
    print(f"\n{'='*90}")
    print(f"{'Batch':<12} {'FP32':>10} {'PTQ W8':>10} {'PCR W8':>10} {'PTQ W8A8':>10} {'PCR W8A8':>10}")
    print(f"{'='*90}")
    for row in rows:
        print(f"{row['batch']:<12} "
              f"{row['fp32']['MAPE']*100:>9.4f}% "
              f"{row['ptq_w8']['MAPE']*100:>9.4f}% "
              f"{row['pcr_w8']['MAPE']*100:>9.4f}% "
              f"{row['ptq_w8a8']['MAPE']*100:>9.4f}% "
              f"{row['pcr_w8a8']['MAPE']*100:>9.4f}%")

    os.makedirs('results', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({'settings': vars(args), 'rows': rows}, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == '__main__':
    main()
