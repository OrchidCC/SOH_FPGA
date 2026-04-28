"""
Train all datasets (TJU, MIT, HUST) × 10 runs each, save to models/checkpoints/.
XJTU already done. Evaluates FP32 MAPE for all datasets after training.
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
from models.Model import PINN, Solution_u
from utils.util import eval_metrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─── Data loaders (copied from main_*.py) ──────────────────────────

def load_xjtu_data(batch_name, args):
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


def load_tju_data(batch_idx, args):
    root = 'data/TJU'
    data = TJUdata(root=root, args=args)
    mod = [(5, 9), (4, 8), (5, 9)]
    batchs = sorted(os.listdir(root))
    batch = batchs[batch_idx]
    batch_root = os.path.join(root, batch)
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
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3'],
    }


def load_mit_data(args):
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
    return {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3'],
    }


def load_hust_data(args):
    test_id = ['1-4', '1-8', '2-4', '2-8',
               '3-4', '3-8', '4-4', '4-8',
               '5-4', '5-7', '6-4', '6-8',
               '7-4', '7-8', '8-4', '8-8',
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
    return {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3'],
    }


# ─── Training ──────────────────────────────────────────────────────

def train_one(args, dataloader, save_folder):
    if os.path.exists(os.path.join(save_folder, 'model.pth')):
        print(f"  [skip] {save_folder} already exists")
        return
    os.makedirs(save_folder, exist_ok=True)
    setattr(args, 'save_folder', save_folder)
    setattr(args, 'log_dir', 'logging.txt')
    pinn = PINN(args)
    pinn.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test'],
    )


# ─── Evaluate FP32 ─────────────────────────────────────────────────

def evaluate_fp32(model_path, testloader):
    model = Solution_u().to('cpu')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['solution_u'])
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for x, _, y, _ in testloader:
            u = model(x.to('cpu'))
            true_labels.append(y.numpy())
            pred_labels.append(u.cpu().numpy())
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_labels, true_labels)
    return {'MAE': MAE, 'MAPE': MAPE * 100, 'MSE': MSE, 'RMSE': RMSE}


def evaluate_dataset(model_dir, testloader, n_runs=10):
    mapes = []
    for e in range(1, n_runs + 1):
        mp = os.path.join(model_dir, f'Experiment{e}', 'model.pth')
        if not os.path.exists(mp):
            continue
        res = evaluate_fp32(mp, testloader)
        mapes.append(res['MAPE'])
    return mapes


# ─── Main ──────────────────────────────────────────────────────────

def main():
    results = {}

    # ── TJU: 3 batches × 10 runs ──
    print("\n" + "=" * 60)
    print("Training TJU (3 batches × 10 runs)")
    print("=" * 60)
    tju_args = argparse.Namespace(
        data='TJU', in_same_batch=True, batch=0, batch_size=512,
        normalization_method='min-max',
        epochs=200, early_stop=20, warmup_epochs=30,
        warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.001,
        F_layers_num=3, F_hidden_dim=60,
        alpha=1, beta=0.05,
        log_dir='logging.txt', save_folder=None,
    )
    tju_batch_names = ['Dataset_1_NCA', 'Dataset_2_NCM', 'Dataset_3_NCM_NCA']
    for bi in range(3):
        tju_args.batch = bi
        dataloader = load_tju_data(bi, tju_args)
        for e in range(1, 11):
            save_dir = f'models/checkpoints/TJU/{bi}-{bi}/Experiment{e}'
            print(f"  TJU batch={bi} run={e}")
            train_one(tju_args, dataloader, save_dir)
        # Evaluate
        mapes = evaluate_dataset(f'models/checkpoints/TJU/{bi}-{bi}', dataloader['test'])
        if mapes:
            print(f"  TJU batch {bi} ({tju_batch_names[bi]}): MAPE = {np.mean(mapes):.4f}% ± {np.std(mapes):.4f}%")
            results[f'TJU_{bi}'] = {'mean': np.mean(mapes), 'std': np.std(mapes), 'n': len(mapes)}

    # ── MIT: 1 × 10 runs ──
    print("\n" + "=" * 60)
    print("Training MIT (10 runs)")
    print("=" * 60)
    mit_args = argparse.Namespace(
        data='MIT', batch_size=512,
        normalization_method='min-max',
        epochs=200, early_stop=10, warmup_epochs=30,
        warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.001,
        F_layers_num=3, F_hidden_dim=60,
        alpha=1, beta=0.02,
        log_dir='logging.txt', save_folder=None,
    )
    mit_data = load_mit_data(mit_args)
    for e in range(1, 11):
        save_dir = f'models/checkpoints/MIT/Experiment{e}'
        print(f"  MIT run={e}")
        train_one(mit_args, mit_data, save_dir)
    mapes = evaluate_dataset('models/checkpoints/MIT', mit_data['test'])
    if mapes:
        print(f"  MIT: MAPE = {np.mean(mapes):.4f}% ± {np.std(mapes):.4f}%")
        results['MIT'] = {'mean': np.mean(mapes), 'std': np.std(mapes), 'n': len(mapes)}

    # ── HUST: 1 × 10 runs ──
    print("\n" + "=" * 60)
    print("Training HUST (10 runs)")
    print("=" * 60)
    hust_args = argparse.Namespace(
        data='HUST', batch_size=512,
        normalization_method='min-max',
        epochs=200, early_stop=20, warmup_epochs=30,
        warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.0005,
        F_layers_num=3, F_hidden_dim=60,
        alpha=0.5, beta=0.2,
        log_dir='logging.txt', save_folder=None,
    )
    hust_data = load_hust_data(hust_args)
    for e in range(1, 11):
        save_dir = f'models/checkpoints/HUST/Experiment{e}'
        print(f"  HUST run={e}")
        train_one(hust_args, hust_data, save_dir)
    mapes = evaluate_dataset('models/checkpoints/HUST', hust_data['test'])
    if mapes:
        print(f"  HUST: MAPE = {np.mean(mapes):.4f}% ± {np.std(mapes):.4f}%")
        results['HUST'] = {'mean': np.mean(mapes), 'std': np.std(mapes), 'n': len(mapes)}

    # ── XJTU (already trained, just evaluate) ──
    print("\n" + "=" * 60)
    print("Evaluating XJTU (already trained)")
    print("=" * 60)
    xjtu_args = argparse.Namespace(
        data='XJTU', batch='2C', batch_size=256,
        normalization_method='min-max',
        log_dir=None, save_folder=None,
    )
    xjtu_batches = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    for i, bn in enumerate(xjtu_batches):
        xjtu_args.batch = bn
        data = load_xjtu_data(bn, xjtu_args)
        mapes = evaluate_dataset(f'models/checkpoints/XJTU/{i}-{i}', data['test'])
        if mapes:
            print(f"  XJTU {bn}: MAPE = {np.mean(mapes):.4f}% ± {np.std(mapes):.4f}%")
            results[f'XJTU_{bn}'] = {'mean': np.mean(mapes), 'std': np.std(mapes), 'n': len(mapes)}

    # ── Summary ──
    print("\n" + "=" * 60)
    print("FP32 Summary (all datasets)")
    print("=" * 60)
    print(f"{'Dataset':<20s}  {'MAPE':>8s}  {'± std':>8s}  {'n':>3s}")
    print("-" * 45)
    for k, v in results.items():
        print(f"{k:<20s}  {v['mean']:>7.4f}%  {v['std']:>7.4f}%  {v['n']:>3d}")


if __name__ == '__main__':
    main()
