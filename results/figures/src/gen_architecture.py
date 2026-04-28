#!/usr/bin/env python3
"""Fig.3 — PINN accelerator architecture on ZYNQ-7020.

Style matched to DSP packing figure (gen_dsp_drawio.py):
  - Times New Roman, fontSize=9
  - strokeColor=#333, strokeWidth=0.6
  - No shadows, sharp corners on content blocks
  - Clean minimal aesthetic
"""

import xml.etree.ElementTree as ET

CELL_ID = [2]
def nid():
    CELL_ID[0] += 1
    return str(CELL_ID[0])

# ─── Colors — warm fills, neutral borders (matching DSP fig) ────
C_OC_BG  = '#EEF2F7'                        # on-chip fill
C_FC_BG  = '#FFF9EC'                         # off-chip fill
C_RGN_BD = '#666666'                         # region dashed border (neutral)
C_BD     = '#333333'                         # block borders (same as DSP)
C_PE     = '#81C784'; C_PE_BD = '#4CAF50'    # PEs
C_MEM    = '#FFE082'; C_MEM_BD = '#FFA000'   # DRAM
C_ARR    = '#555555'                         # arrows
FN = 'fontFamily=Times New Roman;'
FS = 9  # default font size

# ─── Primitives ───────────────────────────────────────────────────
def cell(r, v, x, y, w, h, s):
    c = ET.SubElement(r, 'mxCell', id=nid(), value=v, style=s,
                      vertex='1', parent='1')
    ET.SubElement(c, 'mxGeometry', x=str(int(x)), y=str(int(y)),
                  width=str(int(w)), height=str(int(h)),
                  **{'as': 'geometry'})

def region(r, x, y, w, h, fill):
    cell(r, '', x, y, w, h,
         f'rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor={fill};'
         f'strokeColor={C_RGN_BD};strokeWidth=1.2;dashed=1;dashPattern=8 4;')

def layer_box(r, x, y, w, h):
    cell(r, '', x, y, w, h,
         f'rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;'
         f'strokeColor={C_BD};strokeWidth=0.6;')

def pe_cell(r, x, y, w=28, h=22):
    cell(r, '<b>PE</b>', x, y, w, h,
         f'rounded=0;whiteSpace=wrap;html=1;fillColor={C_PE};strokeColor={C_PE_BD};'
         f'strokeWidth=0.6;{FN}fontSize=8;fontColor=#FFFFFF;fontStyle=1;'
         f'align=center;verticalAlign=middle;')

def lbl(r, x, y, v, fs=FS, fc='#333', al='center', w=50, h=16):
    cell(r, v, x, y, w, h,
         f'text;html=1;{FN}fontSize={fs};fontColor={fc};'
         f'align={al};verticalAlign=middle;resizable=0;points=[];'
         f'autosize=1;strokeColor=none;fillColor=none;')

def vlbl(r, cx, cy, v, fs=9):
    cell(r, f'<b>{v}</b>', cx - 35, cy - 10, 70, 20,
         f'text;html=1;{FN}fontSize={fs};fontColor={C_RGN_BD};fontStyle=1;'
         f'align=center;verticalAlign=middle;rotation=-90;'
         f'resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;')

def arr(r, x1, y1, x2, y2, v='', dash=0, sw=1.0):
    d = 'dashed=1;dashPattern=6 3;' if dash else ''
    c = ET.SubElement(r, 'mxCell', id=nid(), value=v,
                      style=f'endArrow=classic;html=1;strokeColor={C_ARR};'
                            f'strokeWidth={sw};endSize=4;{d}{FN}fontSize=8;'
                            f'fontColor=#555;',
                      edge='1', parent='1')
    g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
    ET.SubElement(g, 'mxPoint', x=str(int(x1)), y=str(int(y1)),
                  **{'as': 'sourcePoint'})
    ET.SubElement(g, 'mxPoint', x=str(int(x2)), y=str(int(y2)),
                  **{'as': 'targetPoint'})

# ═══════════════════════════════════════════════════════════════════
mxfile = ET.Element('mxfile')
diag = ET.SubElement(mxfile, 'diagram', name='Architecture')
mdl = ET.SubElement(diag, 'mxGraphModel', dx='0', dy='0',
                     grid='1', gridSize='4', guides='1')
root = ET.SubElement(mdl, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')

# ═══════════════════════════════════════════════════════════════════
# Layout — compact
# ═══════════════════════════════════════════════════════════════════
LAY_W = 70; LAY_GAP = 12
PE_W = 28; PE_H = 22; PE_GAP = 4
PE_MX = (LAY_W - PE_W) // 2

LAYERS = [
    ('Layer 0', 10, 3),
    ('Layer 1', 20, 4),
    ('Layer 2', 11, 3),
    ('Layer 3', 11, 3),
    ('Layer 4',  1, 1),
]

def layer_height(n_rows, has_dots):
    h = 22
    h += n_rows * PE_H + max(0, n_rows - 1) * PE_GAP
    if has_dots: h += 14
    h += 8
    return h

max_h = max(layer_height(nr, np > nr) for _, np, nr in LAYERS)
LAY_H = max_h
LAY_Y = 16

OC_X = 28; OC_Y = 0
OC_W = 484
OC_H = LAY_Y + LAY_H + 16

FC_Y = OC_H + 32
FC_X = OC_X; FC_W = OC_W; FC_H = 44

PIPE_TOTAL = len(LAYERS) * LAY_W + (len(LAYERS) - 1) * LAY_GAP
PIPE_X0 = OC_X + (OC_W - PIPE_TOTAL) // 2

# ═══════════════════════════════════════════════════════════════════
# 1. Regions
# ═══════════════════════════════════════════════════════════════════
region(root, OC_X, OC_Y, OC_W, OC_H, C_OC_BG)
region(root, FC_X, FC_Y, FC_W, FC_H, C_FC_BG)
vlbl(root, OC_X + 4, OC_Y + OC_H // 2, 'On-chip')
vlbl(root, FC_X + 4, FC_Y + FC_H // 2, 'Off-chip')

# ═══════════════════════════════════════════════════════════════════
# 2. Layers
# ═══════════════════════════════════════════════════════════════════
layer_pos = []

for i, (name, n_pe, n_rows) in enumerate(LAYERS):
    lx = PIPE_X0 + i * (LAY_W + LAY_GAP)
    layer_pos.append((lx, LAY_W))

    layer_box(root, lx, LAY_Y, LAY_W, LAY_H)

    lbl(root, lx, LAY_Y + 3, f'<b>{name}</b>',
        fs=FS, fc='#333', al='center', w=LAY_W, h=14)

    pe_y0 = LAY_Y + 20
    for j in range(n_rows):
        pe_cell(root, lx + PE_MX, pe_y0 + j * (PE_H + PE_GAP))

    if n_pe > n_rows:
        dy = pe_y0 + n_rows * (PE_H + PE_GAP) - 1
        lbl(root, lx, dy, '⋮', fs=11, fc='#999', al='center', w=LAY_W, h=12)

# ═══════════════════════════════════════════════════════════════════
# 3. Pipeline arrows
# ═══════════════════════════════════════════════════════════════════
arr_y = LAY_Y + LAY_H // 2
for i in range(len(LAYERS) - 1):
    x1 = layer_pos[i][0] + layer_pos[i][1]
    x2 = layer_pos[i + 1][0]
    arr(root, x1, arr_y, x2, arr_y)

# ═══════════════════════════════════════════════════════════════════
# 4. Main Memory
# ═══════════════════════════════════════════════════════════════════
MM_PAD = 50
MM_X = FC_X + MM_PAD; MM_Y = FC_Y + 7; MM_W = FC_W - 2 * MM_PAD; MM_H = 30
cell(root, '<b>Main Memory (DRAM)</b>', MM_X, MM_Y, MM_W, MM_H,
     f'rounded=0;whiteSpace=wrap;html=1;fillColor={C_MEM};strokeColor={C_MEM_BD};'
     f'strokeWidth=0.6;{FN}fontSize={FS};align=center;verticalAlign=middle;')

# ═══════════════════════════════════════════════════════════════════
# 5. Features → L0, L4 → Output
# ═══════════════════════════════════════════════════════════════════
MID_Y = (OC_H + FC_Y) // 2

l0_cx = layer_pos[0][0] + layer_pos[0][1] // 2
l0_bot = LAY_Y + LAY_H
lbl(root, l0_cx - 28, MID_Y - 7, '<b>Features</b>',
    fs=8, fc='#555', al='center', w=56, h=14)
arr(root, l0_cx, MM_Y, l0_cx, MID_Y + 8, dash=1, sw=0.8)
arr(root, l0_cx, MID_Y - 10, l0_cx, l0_bot, dash=1, sw=0.8)

l4_cx = layer_pos[-1][0] + layer_pos[-1][1] // 2
l4_bot = LAY_Y + LAY_H
lbl(root, l4_cx - 22, MID_Y - 7, '<b>Output</b>',
    fs=8, fc='#555', al='center', w=44, h=14)
arr(root, l4_cx, l4_bot, l4_cx, MID_Y - 10, dash=1, sw=0.8)
arr(root, l4_cx, MID_Y + 8, l4_cx, MM_Y, dash=1, sw=0.8)

# ═══════════════════════════════════════════════════════════════════
tree = ET.ElementTree(mxfile)
ET.indent(tree, space='  ')
out = '/home/orchid/Desktop/Project/PINN_FPGA/results/figures/src/fig_architecture.drawio'
tree.write(out, xml_declaration=True, encoding='UTF-8')
print(f'Saved: {out}')
