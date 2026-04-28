#!/usr/bin/env python3
"""Fig.6: Pipeline timing Gantt chart — shows streaming execution of consecutive samples."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import os
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

def setup_fonts():
    available = [f.name for f in fm.fontManager.ttflist]
    font_name = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [font_name, 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })

def main():
    setup_fonts()

    # Per-layer cycle counts (from CLAUDE.md PE Allocation)
    layers = ['Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    cycles = [34, 60, 60, 32, 32]
    n_layers = len(layers)
    n_samples = 4  # show 4 samples flowing through

    # Colors for each sample
    colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']
    alpha = 0.85

    fig, ax = plt.subplots(figsize=(5.5, 2.0))

    bar_height = 0.6

    for s in range(n_samples):
        for i in range(n_layers):
            # Start time: sample must wait for previous layer to finish,
            # AND previous sample must have cleared this layer
            if s == 0 and i == 0:
                start = 0
            elif i == 0:
                # First layer starts after bottleneck interval
                start = s * max(cycles)
            else:
                # Must wait for: (1) this sample's previous layer, (2) previous sample's this layer
                prev_layer_end = starts[i-1] + cycles[i-1]
                if s > 0:
                    prev_sample_end = prev_starts[i] + cycles[i]
                    start = max(prev_layer_end, prev_sample_end)
                else:
                    start = prev_layer_end

            if i == 0:
                starts = [start]
            else:
                starts.append(start)

            ax.barh(n_layers - 1 - i, cycles[i], left=start, height=bar_height,
                    color=colors[s], alpha=alpha, edgecolor='#333', linewidth=0.5)

            # Label sample number inside bar (only if wide enough)
            if cycles[i] >= 25:
                ax.text(start + cycles[i]/2, n_layers - 1 - i,
                        f'S{s+1}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

        prev_starts = starts.copy()

    # Bottleneck annotation
    bottleneck_period = max(cycles)
    for s in range(1, n_samples):
        x = s * bottleneck_period
        ax.axvline(x, color='#999', linestyle='--', linewidth=0.5, zorder=0)

    # Throughput annotation (above Layer 0)
    ann_y = n_layers - 0.5 + 0.4
    ax.annotate('', xy=(bottleneck_period, ann_y), xytext=(0, ann_y),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=0.8))
    ax.text(bottleneck_period/2, ann_y + 0.15, f'{bottleneck_period} cycles', ha='center', va='bottom', fontsize=8, color='#333')

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(reversed(layers))
    ax.set_xlabel('Clock cycles')
    # Compute actual end time
    max_end = max(starts[i] + cycles[i] for i in range(n_layers))
    ax.set_xlim(-5, max_end + 15)
    ax.set_ylim(-0.5, n_layers - 0.5 + 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')

    # Legend
    patches = [mpatches.Patch(color=colors[i], alpha=alpha, label=f'Sample {i+1}') for i in range(n_samples)]
    ax.legend(handles=patches, loc='upper right', fontsize=7, frameon=False, ncol=2)

    plt.tight_layout(pad=0.3)

    pdf_path = os.path.join(OUT_DIR, 'fig_pipeline_gantt.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {pdf_path}")

if __name__ == '__main__':
    main()
