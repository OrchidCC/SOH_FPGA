#!/usr/bin/env python3
"""Fig.1 (Background): Battery capacity degradation — HUST dataset, mean+envelope."""

import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'HUST')
OUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures', 'output')
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_fonts():
    available = [f.name for f in fm.fontManager.ttflist]
    font_name = 'Times New Roman' if 'Times New Roman' in available else 'DejaVu Serif'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [font_name, 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })


def load_batch(batch_files, max_cycles=None):
    """Load batteries, trim to min length, return envelope."""
    all_caps = []
    for f in batch_files:
        df = pd.read_csv(os.path.join(DATA_ROOT, f))
        all_caps.append(df['capacity'].values)
    min_len = min(len(c) for c in all_caps)
    if max_cycles:
        min_len = min(min_len, max_cycles)
    trimmed = np.array([c[:min_len] for c in all_caps])
    cycles = np.arange(min_len)
    from scipy.ndimage import uniform_filter1d
    mean = uniform_filter1d(trimmed.mean(axis=0), size=10)
    std = uniform_filter1d(trimmed.std(axis=0), size=10)
    lo = mean - std
    hi = mean + std
    return cycles, mean, lo, hi


def main():
    setup_fonts()

    # Group files by batch
    files = sorted([f for f in os.listdir(DATA_ROOT) if f.endswith('.csv')])
    batches = {}
    for f in files:
        b = f.split('-')[0]
        batches.setdefault(b, []).append(f)

    # 2 contrasting batches: fast vs slow degradation
    pick = ['1', '9']
    colors = ['#D6604D', '#2166AC']
    labels = ['Fast degradation', 'Slow degradation']

    fig, ax = plt.subplots(figsize=(4, 2.8))

    for b, color, label in zip(pick, colors, labels):
        if b not in batches:
            continue
        cycles, mean, lo, hi = load_batch(batches[b], max_cycles=1123)
        ax.fill_between(cycles, lo, hi, color=color, alpha=0.15)
        ax.plot(cycles, mean, color=color, linewidth=1.0, alpha=0.85, label=label)

    ax.set_xlabel('Cycle')
    ax.set_ylabel('Capacity (Ah)')
    ax.legend(frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')

    plt.tight_layout(pad=0.3)

    pdf_path = os.path.join(OUT_DIR, 'fig_background.pdf')
    png_path = os.path.join(OUT_DIR, 'fig_background.png')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
