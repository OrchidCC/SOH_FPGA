#!/usr/bin/env python3
"""Fig. 6: channel-specific LUT mappings vs shared per-layer LUT."""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def setup_fonts():
    times_path = '/usr/share/fonts/truetype/msttcorefonts/Times.TTF'
    font_name = 'DejaVu Serif'
    if os.path.exists(times_path):
        fm.fontManager.addfont(times_path)
        font_name = fm.FontProperties(fname=times_path).get_name()
    elif any(f.name == 'Nimbus Roman' for f in fm.fontManager.ttflist):
        font_name = 'Nimbus Roman'
    plt.rcParams.update({
        'font.family': font_name,
        'font.serif': [font_name, 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 7.0,
        'axes.labelsize': 7.0,
        'xtick.labelsize': 6.0,
        'ytick.labelsize': 6.0,
        'legend.fontsize': 5.0,
        'axes.linewidth': 0.55,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })


def compute_channel_mapping(scale, bias, acc_scale, k=4):
    """Return the 16-entry channel mapping indexed by quantized input code."""
    qm = 2 ** (k - 1) - 1
    q_inputs = np.arange(-2 ** (k - 1), 2 ** (k - 1))  # -8..7
    x_real = q_inputs * acc_scale
    y_sin = np.sin(x_real)
    centered = (y_sin - bias) / scale
    q_out = np.clip(np.round(centered * qm), -qm, qm)
    y_lut = q_out / qm * scale + bias
    return q_inputs, y_lut


def main():
    setup_fonts()

    # Two representative channels with distinct learned mappings.
    q_a, y_a = compute_channel_mapping(scale=0.24, bias=0.00, acc_scale=0.035)
    q_b, y_b = compute_channel_mapping(scale=0.95, bias=-0.10, acc_scale=0.16)

    # Shared per-layer baseline: one mapping must be reused for both channels.
    q_s, y_s = compute_channel_mapping(scale=0.595, bias=-0.05, acc_scale=0.10)

    fig, ax = plt.subplots(figsize=(2.58, 1.74))

    ax.step(q_a, y_a, where='mid', color='#4E79A7', linewidth=0.95, label='Ch. A')
    ax.plot(q_a, y_a, 'o', color='#4E79A7', markersize=1.8)

    ax.step(q_b, y_b, where='mid', color='#E15759', linewidth=0.95, label='Ch. B')
    ax.plot(q_b, y_b, 's', color='#E15759', markersize=1.8)

    ax.step(q_s, y_s, where='mid', color='#F28E2B', linewidth=0.85, linestyle='--',
            label='Shared')

    ax.set_xlabel('Quantized input level')
    ax.set_ylabel('LUT output')
    ax.set_xlim(-8.5, 7.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-8, -4, 0, 4, 7])
    ax.grid(axis='y', alpha=0.12, linewidth=0.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', width=0.5, length=2.2, pad=1.5)
    ax.legend(frameon=False, loc='upper left', handlelength=1.3,
              borderpad=0.1, labelspacing=0.2, handletextpad=0.4)

    plt.tight_layout(pad=0.18)

    pdf_path = os.path.join(OUT_DIR, 'fig_lut_viz.pdf')
    png_path = os.path.join(OUT_DIR, 'fig_lut_viz.png')
    fig.savefig(pdf_path, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


if __name__ == '__main__':
    main()
