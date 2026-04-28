#!/usr/bin/env python3
"""Fig.2: Per-channel GridWarp motivation.

Single-panel box plot: 60 channels from Sin layer 1, sorted by median.
Shows channel-dependent distribution diversity → motivates per-channel quantization.

Usage:
    cd /home/orchid/Desktop/Project/PINN_FPGA
    python3 results/figures/src/gen_gridwarp_fig.py
"""

import sys, os
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.Model import Solution_u, Sin
from utils.dataloader import XJTUdata
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'checkpoints', 'XJTU', '0-0',
                          'Experiment1', 'model.pth')
DATA_ROOT  = os.path.join(PROJECT_ROOT, 'data', 'XJTU')
OUT_DIR    = os.path.join(PROJECT_ROOT, 'results', 'figures', 'output')
os.makedirs(OUT_DIR, exist_ok=True)
device = 'cpu'


def setup_fonts():
    available = [f.name for f in fm.fontManager.ttflist]
    font_name = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [font_name, 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 8,
        'axes.linewidth': 0.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })


def load_model_and_data():
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = Solution_u()
    model.load_state_dict(ckpt['solution_u'])
    model.eval().to(device)
    args = argparse.Namespace(
        data='XJTU', batch='2C', batch_size=256,
        normalization_method='min-max', log_dir=None, save_folder=None)
    data = XJTUdata(root=DATA_ROOT, args=args)
    loaders = data.read_one_batch('2C')
    all_x = torch.cat([x for x, _, _, _ in loaders['train_2']])
    return model, all_x


def collect_sin1_outputs(model, x):
    """Forward pass, collect first sin layer outputs (60 channels)."""
    h = x.detach()
    with torch.no_grad():
        for mod in list(model.encoder.net) + list(model.predictor.net):
            if isinstance(mod, nn.Linear):
                h = mod(h)
            elif isinstance(mod, Sin):
                return torch.sin(h).numpy()  # first sin layer
            elif isinstance(mod, nn.Dropout):
                pass
    return None


def main():
    setup_fonts()
    model, x_calib = load_model_and_data()
    sin1 = collect_sin1_outputs(model, x_calib[:512].to(device))
    n_ch = sin1.shape[1]

    # Sort channels by median
    medians = np.median(sin1, axis=0)
    order = np.argsort(medians)
    sorted_data = [sin1[:, order[i]] for i in range(n_ch)]

    # Two panels: (a) box plot, (b) single channel histogram + grid
    fig, (ax, ax_in) = plt.subplots(1, 2, figsize=(7, 2.5),
                                      gridspec_kw={'width_ratios': [2.2, 1]})

    bp = ax.boxplot(sorted_data, widths=0.6, patch_artist=True,
                    showfliers=False, medianprops=dict(color='#333', linewidth=0.8),
                    whiskerprops=dict(color='#555', linewidth=0.5),
                    capprops=dict(color='#555', linewidth=0.5),
                    boxprops=dict(linewidth=0.4))

    # Color boxes by median value (gradient)
    cmap = plt.cm.RdYlBu_r
    norm_medians = (medians[order] - medians.min()) / (medians.max() - medians.min() + 1e-9)
    for i, (patch, nm) in enumerate(zip(bp['boxes'], norm_medians)):
        patch.set_facecolor(cmap(nm))
        patch.set_alpha(0.7)

    ax.set_xlabel('Channel index')
    ax.set_ylabel('Sin output value')

    ax.set_xticks(range(1, n_ch + 1, 10))
    ax.set_xticklabels(range(0, n_ch, 10))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel (b): one channel histogram + uniform grid lines ──
    overall_median = np.median(medians)
    rep_ch = np.argmin(np.abs(medians - overall_median))
    vals = sin1[:, rep_ch]

    ax_in.hist(vals, bins=25, color='#2A9D8F', alpha=0.7, edgecolor='none', density=True)

    layer_absmax = np.abs(sin1).max()
    qm = 7
    uniform_step = layer_absmax / qm
    for i in range(-8, 8):
        x_grid = i * uniform_step
        ax_in.axvline(x=x_grid, color='#E63946', linewidth=0.9, linestyle='-', alpha=0.6)

    ax_in.set_xlim(-1.05, 1.05)
    ax_in.set_xlabel('Activation value')
    ax_in.set_ylabel('Density')
    ax_in.set_title('(b) Uniform grid vs. actual distribution', fontsize=9)
    ax_in.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_in.spines['top'].set_visible(False)
    ax_in.spines['right'].set_visible(False)

    # Add (a) label to box plot
    ax.set_title('(a) Per-channel sin output distribution', fontsize=9)

    plt.tight_layout(pad=0.4, w_pad=1.5)

    pdf_path = os.path.join(OUT_DIR, 'fig_gridwarp.pdf')
    png_path = os.path.join(OUT_DIR, 'fig_gridwarp.png')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
