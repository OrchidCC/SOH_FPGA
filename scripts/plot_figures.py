#!/usr/bin/env python3
"""Generate all paper figures with unified style."""

import sys, os, json, numpy as np, torch, torch.nn as nn, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.Model import Solution_u, MLP, Sin
from utils.dataloader import XJTUdata

# ─── Unified Style ────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_FP32   = '#2ecc71'
C_PTQ    = '#e74c3c'
C_QAT    = '#3498db'
C_LGW    = '#f39c12'
C_CGW    = '#9b59b6'
C_FAILED = '#bdc3c7'

# Neutral colors for data distributions (never overlap with method colors)
C_DATA1  = '#5d6d7e'  # dark slate
C_DATA2  = '#85929e'  # medium slate
C_DATA3  = '#aeb6bf'  # light slate
C_DATA4  = '#d5dbdb'  # very light
# Channel example colors (teal/cyan palette, no overlap with method colors)
C_CH = ['#1abc9c', '#16a085', '#148f77', '#117a65']

os.makedirs('results/figures', exist_ok=True)

# ─── Load data ─────────────────────────────────────────────────────

def load_ablation():
    with open('results/w4a4_ablation_final.json') as f:
        return json.load(f)

def load_fp32_model(batch='2C'):
    """Load one FP32 model for distribution analysis."""
    mp = 'models/checkpoints/XJTU/0-0/Experiment1/model.pth'
    ckpt = torch.load(mp, map_location='cpu', weights_only=False)
    model = Solution_u()
    model.load_state_dict(ckpt['solution_u'])
    model.eval()
    return model

def load_calib_data(batch='2C'):
    args = argparse.Namespace(data='XJTU', batch=batch, batch_size=256,
                               normalization_method='min-max', log_dir=None, save_folder=None)
    root = 'data/XJTU'
    data = XJTUdata(root=root, args=args)
    tl = [os.path.join(root, f) for f in os.listdir(root)
          if batch in f and '4' not in f and '8' not in f]
    tr = data.read_all(specific_path_list=tl)
    x_all = torch.cat([x for x, _, _, _ in tr['train_2']])
    return x_all[:512]

def get_layer_activations(model, x):
    """Get sin activations and pre-activations for each layer."""
    pre_acts, post_acts = [], []
    h = x.detach()
    for mod in list(model.encoder.net) + list(model.predictor.net):
        if isinstance(mod, nn.Linear):
            with torch.no_grad(): h = mod(h)
            pre_acts.append(h.clone())
        elif isinstance(mod, Sin):
            with torch.no_grad(): h = torch.sin(h)
            post_acts.append(h.clone())
        elif isinstance(mod, nn.Dropout):
            pass
    return pre_acts, post_acts

def get_weight_stats(model):
    """Get per-row weight statistics."""
    stats = []
    li = 0
    for mod in list(model.encoder.net) + list(model.predictor.net):
        if isinstance(mod, nn.Linear):
            w = mod.weight.detach()
            row_means = w.mean(dim=1).numpy()
            row_stds = w.std(dim=1).numpy()
            row_max = w.abs().amax(dim=1).numpy()
            stats.append({'name': f'L{li}', 'shape': tuple(w.shape),
                          'means': row_means, 'stds': row_stds, 'max': row_max,
                          'flat': w.numpy().flatten()})
            li += 1
    return stats


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Motivation — Weight vs Activation distribution mismatch
# ═══════════════════════════════════════════════════════════════════

def fig1_motivation(model, x_calib):
    """Show WHY weight-side methods fail and activation-side succeeds.
    Left: weight rows are symmetric, well-distributed → no mismatch → methods fail
    Right: sin activation channels are non-uniform → mismatch → GridWarp fixes it
    """
    w_stats = get_weight_stats(model)
    _, post_acts = get_layer_activations(model, x_calib)

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Top-left: L3 weight distribution (per-row) ──
    ax = fig.add_subplot(gs[0, 0])
    w3 = w_stats[3]  # L3
    # Show weight histogram
    ax.hist(w3['flat'], bins=50, color=C_DATA1, alpha=0.7, edgecolor='white', linewidth=0.3)
    # Overlay W4 grid
    wmax = np.abs(w3['flat']).max()
    grid = np.linspace(-wmax, wmax, 15)
    for g in grid:
        ax.axvline(x=g, color=C_DATA3, alpha=0.6, linewidth=0.8, linestyle='--')
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Count')
    ax.set_title('(a) L3 weight distribution + W4 grid')
    ax.annotate('Symmetric, well-covered\n→ No mismatch',
                xy=(0.02, 0.95), xycoords='axes fraction', va='top',
                fontsize=9, color='#27ae60', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Top-right: L3 weight per-row means (all near 0) ──
    ax = fig.add_subplot(gs[0, 1])
    rows = np.arange(len(w3['means']))
    ax.bar(rows, w3['means'], color=C_DATA2, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Row index')
    ax.set_ylabel('Row mean')
    ax.set_title('(b) L3 per-row weight mean')
    ax.annotate(f'All near zero (max |mean|={np.abs(w3["means"]).max():.3f})\n→ Weight GridWarp bias useless',
                xy=(0.02, 0.95), xycoords='axes fraction', va='top',
                fontsize=9, color='#e74c3c', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Bottom-left: sin3 per-channel distribution (box plot) ──
    ax = fig.add_subplot(gs[1, 0])
    sin3 = post_acts[2].numpy()  # (N, 32)
    bp = ax.boxplot([sin3[:, i] for i in range(32)], patch_artist=True,
                     showfliers=False, medianprops=dict(color='white', linewidth=1.2),
                     whiskerprops=dict(color='#555'), capprops=dict(color='#555'))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(C_DATA1)
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Channel index')
    ax.set_ylabel('sin(z) output value')
    ax.set_title('(c) Sin3 per-channel activation distribution')
    ax.annotate('Channels have different\nscale and offset\n→ Per-channel mismatch!',
                xy=(0.98, 0.95), xycoords='axes fraction', va='top', ha='right',
                fontsize=9, color='#8e44ad', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # ── Bottom-right: 4 example channels vs uniform A4 grid ──
    ax = fig.add_subplot(gs[1, 1])
    chs = [0, 8, 16, 24]
    for ch, cc in zip(chs, C_CH):
        vals = sin3[:, ch]
        ax.hist(vals, bins=30, alpha=0.45, color=cc, label=f'Ch {ch}', density=True,
                edgecolor='white', linewidth=0.3)
    # Uniform A4 grid (per-layer: uses global max)
    layer_max = np.abs(sin3).max()
    grid = np.linspace(-layer_max, layer_max, 15)
    for g in grid:
        ax.axvline(x=g, color=C_DATA3, alpha=0.6, linewidth=0.8)
    ax.set_xlabel('Activation value')
    ax.set_ylabel('Density')
    ax.set_title('(d) Per-layer A4 grid (gray) vs channel distributions')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.annotate('Outer grid lines\nwaste resolution\non narrow channels',
                xy=(0.02, 0.95), xycoords='axes fraction', va='top',
                fontsize=9, color='#e74c3c', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('Figure 1: Weight vs Activation Distribution Analysis — Motivation for Per-channel GridWarp',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig('results/figures/fig1_motivation.pdf', bbox_inches='tight')
    plt.savefig('results/figures/fig1_motivation.png', bbox_inches='tight')
    print("  Saved: fig1_motivation.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Ablation bar chart
# ═══════════════════════════════════════════════════════════════════

def fig2_ablation(all_results):
    datasets = list(all_results.keys())
    short_names = [ds.replace('XJTU_','X-').replace('TJU_','T-') for ds in datasets]

    methods = ['FP32', 'PTQ', 'QAT', 'LGW', 'CGW']
    labels = ['FP32', 'PTQ', 'QAT', '+Layer GW', '+Channel GW']
    colors = [C_FP32, C_PTQ, C_QAT, C_LGW, C_CGW]

    n_ds = len(datasets)
    n_m = len(methods)
    x = np.arange(n_ds)
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        key = 'fp32' if method == 'FP32' else method
        vals = [all_results[ds]['mean'][key] for ds in datasets]
        errs = [all_results[ds]['std'].get(key, 0) for ds in datasets]
        bars = ax.bar(x + (i - n_m/2 + 0.5)*width, vals, width, yerr=errs,
                      label=label, color=color, capsize=2, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('MAPE (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.legend(fontsize=10, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.12),
              framealpha=0.9)
    ax.set_ylim(0, 8)

    # Add average line
    avg_cgw = np.mean([all_results[ds]['mean']['CGW'] for ds in datasets])
    ax.axhline(y=avg_cgw, color=C_CGW, linestyle='--', alpha=0.6, linewidth=1)
    ax.annotate(f'Avg +CGW: {avg_cgw:.2f}%', xy=(n_ds-0.5, avg_cgw+0.15),
                color=C_CGW, fontsize=9, fontweight='bold')

    ax.set_title('Figure 2: W4A4 Ablation — Each Component\'s Contribution', fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/fig2_ablation.pdf', bbox_inches='tight')
    plt.savefig('results/figures/fig2_ablation.png', bbox_inches='tight')
    print("  Saved: fig2_ablation.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Step-by-step improvement waterfall
# ═══════════════════════════════════════════════════════════════════

def fig3_waterfall(all_results):
    """Waterfall chart showing each step's contribution."""
    datasets = list(all_results.keys())

    # Average MAPE per method
    avg = {k: np.mean([all_results[ds]['mean'][k] for ds in datasets])
           for k in ['PTQ', 'QAT', 'LGW', 'CGW']}

    steps = ['PTQ\n(baseline)', '+QAT', '+Layer\nGridWarp', '+Channel\nGridWarp']
    values = [avg['PTQ'], avg['QAT'], avg['LGW'], avg['CGW']]
    deltas = [0, values[1]-values[0], values[2]-values[1], values[3]-values[2]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: waterfall
    ax = axes[0]
    colors_bar = [C_PTQ, C_QAT, C_LGW, C_CGW]
    bottoms = [0, values[0], values[1], values[2]]

    # Draw the bars showing final MAPE at each step
    for i in range(4):
        ax.bar(i, values[i], color=colors_bar[i], alpha=0.8, edgecolor='white', width=0.6)
        ax.text(i, values[i] + 0.1, f'{values[i]:.2f}%', ha='center', fontsize=10, fontweight='bold')

    # Draw arrows for deltas
    for i in range(1, 4):
        ax.annotate('', xy=(i, values[i]), xytext=(i-1, values[i-1]),
                     arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
        mid_y = (values[i-1] + values[i]) / 2
        ax.text(i - 0.5, mid_y, f'{deltas[i]:+.2f}%', ha='center', fontsize=9,
                color='#27ae60' if deltas[i] < 0 else '#e74c3c', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))

    ax.set_xticks(range(4))
    ax.set_xticklabels(steps)
    ax.set_ylabel('MAPE (%)')
    ax.set_ylim(0, 7)
    ax.set_title('(a) Average MAPE: step-by-step improvement')

    # Right: per-dataset improvement (CGW vs LGW)
    ax = axes[1]
    improvements = [all_results[ds]['mean']['LGW'] - all_results[ds]['mean']['CGW']
                    for ds in datasets]
    short_names = [ds.replace('XJTU_','X-').replace('TJU_','T-') for ds in datasets]
    bars = ax.barh(range(len(datasets)), improvements, color=C_CGW, alpha=0.8,
                    edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(short_names)
    ax.set_xlabel('MAPE improvement (%)')
    ax.set_title('(b) Per-channel vs Per-layer GridWarp improvement')
    ax.axvline(x=0, color='black', linewidth=0.5)
    for i, v in enumerate(improvements):
        ax.text(v + 0.02, i, f'{v:.2f}%', va='center', fontsize=9)

    avg_imp = np.mean(improvements)
    ax.axvline(x=avg_imp, color=C_CGW, linestyle='--', alpha=0.6)
    ax.text(avg_imp + 0.02, len(datasets) - 0.5, f'Avg: {avg_imp:.2f}%',
            color=C_CGW, fontsize=9, fontweight='bold')

    fig.suptitle('Figure 3: Quantization Method Progression and Per-channel GridWarp Gains',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/figures/fig3_waterfall.pdf', bbox_inches='tight')
    plt.savefig('results/figures/fig3_waterfall.png', bbox_inches='tight')
    print("  Saved: fig3_waterfall.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Per-channel vs per-layer quantization grid illustration
# ═══════════════════════════════════════════════════════════════════

def fig4_grid_comparison(model, x_calib):
    """Illustrate how per-channel GridWarp adapts the A4 grid per channel."""
    _, post_acts = get_layer_activations(model, x_calib)
    sin3 = post_acts[2].numpy()  # (N, 32)

    # Pick 6 channels with diverse distributions
    ch_stds = sin3.std(axis=0)
    ch_means = sin3.mean(axis=0)
    # Sort by std to get diverse ones
    sorted_idx = np.argsort(ch_stds)
    pick = [sorted_idx[0], sorted_idx[5], sorted_idx[10],
            sorted_idx[20], sorted_idx[25], sorted_idx[31]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=False)

    layer_max = np.abs(sin3).max()
    layer_grid = np.linspace(-layer_max, layer_max, 15)

    for idx, (ch, ax) in enumerate(zip(pick, axes.flatten())):
        vals = sin3[:, ch]
        ch_max = np.quantile(np.abs(vals), 0.995)
        ch_mean = vals.mean()
        ch_grid = np.linspace(ch_mean - ch_max, ch_mean + ch_max, 15)

        ax.hist(vals, bins=30, alpha=0.5, color=C_DATA1, density=True,
                edgecolor='white', linewidth=0.3, label='Distribution')

        # Per-layer grid (light gray, solid)
        for g in layer_grid:
            ax.axvline(x=g, color=C_DATA3, alpha=0.5, linewidth=0.8)
        # Per-channel grid (purple, dashed)
        for g in ch_grid:
            ax.axvline(x=g, color=C_CGW, alpha=0.7, linewidth=1.2, linestyle='--')

        ax.set_title(f'Ch {ch} (std={ch_stds[ch]:.3f})', fontsize=10)
        if idx >= 3:
            ax.set_xlabel('Value')
        if idx % 3 == 0:
            ax.set_ylabel('Density')

    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_DATA3, linewidth=1, label='Per-layer A4 grid'),
        Line2D([0], [0], color=C_CGW, linewidth=1.2, linestyle='--', label='Per-channel A4 grid'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               bbox_to_anchor=(0.5, 1.03), fontsize=11)

    fig.suptitle('Figure 4: Per-layer vs Per-channel A4 Quantization Grid Adaptation',
                 fontsize=13, fontweight='bold', y=1.08)
    plt.tight_layout()
    plt.savefig('results/figures/fig4_grid_comparison.pdf', bbox_inches='tight')
    plt.savefig('results/figures/fig4_grid_comparison.png', bbox_inches='tight')
    print("  Saved: fig4_grid_comparison.pdf/png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════

def main():
    print("Loading data...")
    all_results = load_ablation()
    model = load_fp32_model()
    x_calib = load_calib_data()

    print("Generating figures...")
    fig1_motivation(model, x_calib)
    fig2_ablation(all_results)
    fig3_waterfall(all_results)
    fig4_grid_comparison(model, x_calib)
    print("\nAll figures saved to results/figures/")


if __name__ == '__main__':
    main()
