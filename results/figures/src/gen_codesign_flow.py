#!/usr/bin/env python3
"""Fig.1 — Co-design flow. Compact layout.

[PINN Training] → [Co-Design: GridWarp QAT → {Per-channel LUT, FPGA Pipeline}] → [SOH]
Algorithm/Hardware labels above blocks, no sub-boxes, tight co-design region.
"""

import xml.etree.ElementTree as ET

CELL_ID = [2]
def nid():
    CELL_ID[0] += 1
    return str(CELL_ID[0])

C_ALG = '#DBEAFE'
C_HW  = '#D1FAE5'
C_BRG = '#FFF3CD'
C_IO  = '#F3F4F6'
C_CO  = '#EDE9FE'
C_BD  = '#333333'
C_ARR = '#555555'
FN = 'fontFamily=Times New Roman;'

def cell(r, v, x, y, w, h, s):
    c = ET.SubElement(r, 'mxCell', id=nid(), value=v, style=s,
                      vertex='1', parent='1')
    ET.SubElement(c, 'mxGeometry', x=str(int(x)), y=str(int(y)),
                  width=str(int(w)), height=str(int(h)),
                  **{'as': 'geometry'})

def box(r, x, y, w, h, lab, fc, fs=9):
    cell(r, lab, x, y, w, h,
         f'rounded=0;whiteSpace=wrap;html=1;fillColor={fc};strokeColor={C_BD};'
         f'strokeWidth=0.6;{FN}fontSize={fs};align=center;verticalAlign=middle;')

def lbl(r, x, y, v, fs=7, fc='#888', al='center', w=80, h=12):
    cell(r, v, x, y, w, h,
         f'text;html=1;{FN}fontSize={fs};fontColor={fc};align={al};'
         f'verticalAlign=middle;resizable=0;points=[];autosize=1;'
         f'strokeColor=none;fillColor=none;')

def arr(r, x1, y1, x2, y2, sw=1.0):
    c = ET.SubElement(r, 'mxCell', id=nid(), value='',
                      style=f'endArrow=classic;html=1;strokeColor={C_ARR};'
                            f'strokeWidth={sw};endSize=4;',
                      edge='1', parent='1')
    g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
    ET.SubElement(g, 'mxPoint', x=str(int(x1)), y=str(int(y1)),
                  **{'as': 'sourcePoint'})
    ET.SubElement(g, 'mxPoint', x=str(int(x2)), y=str(int(y2)),
                  **{'as': 'targetPoint'})

# ═══════════════════════════════════════════════════════════════════
mxfile = ET.Element('mxfile')
diag = ET.SubElement(mxfile, 'diagram', name='Co-design Flow')
mdl = ET.SubElement(diag, 'mxGraphModel', dx='0', dy='0',
                     grid='1', gridSize='4', guides='1')
root = ET.SubElement(mdl, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')

# ═══════════════════════════════════════════════════════════════════
# Sizing — compact
# ═══════════════════════════════════════════════════════════════════
BW = 88; BH = 34; GAP = 14
SM_W = 48; SM_H = 28
BLK_W = 104; BLK_H = 30  # inner blocks
BLK_GAP = 6              # vertical gap between stacked HW blocks
MID_GAP = 12             # gap between algorithm block and hardware blocks
CO_PAD = 8               # padding inside co-design box
LBL_H = 12               # label height above blocks

# Hardware column height
HW_COL_H = 2 * BLK_H + BLK_GAP

# Co-design box: tight around content
# Content: label row + blocks
CO_CONTENT_H = LBL_H + HW_COL_H
CO_W = CO_PAD + BLK_W + MID_GAP + (BLK_W + 12) + CO_PAD
CO_H = CO_PAD + CO_CONTENT_H + CO_PAD

# X positions
x_train = 0
x_co = x_train + BW + GAP
x_soh = x_co + CO_W + GAP

# Y: co-design box at top, everything aligns to it
co_y = 0
content_y = co_y + CO_PAD
lbl_y = content_y          # "Algorithm" / "Hardware" labels
blk_y = lbl_y + LBL_H      # blocks start

main_cy = co_y + CO_H // 2
train_y = main_cy - BH // 2
soh_y = main_cy - SM_H // 2

# ─── PINN Training ───────────────────────────────────────────────
box(root, x_train, train_y, BW, BH, '<b>PINN Training</b>', C_ALG)

# ─── Co-Design dashed region (tight) ─────────────────────────────
cell(root, '', x_co, co_y, CO_W, CO_H,
     f'rounded=0;whiteSpace=wrap;html=1;fillColor={C_CO};strokeColor={C_BD};'
     f'strokeWidth=0.8;dashed=1;dashPattern=6 3;')

# ─── Algorithm block (left) ──────────────────────────────────────
alg_x = x_co + CO_PAD
# "Algorithm" label centered above block
lbl(root, alg_x, lbl_y, '<i>Algorithm</i>', fs=7, fc='#888',
    al='center', w=BLK_W, h=LBL_H)
# GridWarp QAT block — vertically centered with HW column
alg_blk_y = blk_y + (HW_COL_H - BLK_H) // 2
box(root, alg_x, alg_blk_y, BLK_W, BLK_H,
    '<b>Per-channel<br>GridWarp QAT</b>', C_ALG, fs=9)

# ─── Hardware sub-box (right, dashed, wraps LUT + Pipeline) ──────
hw_x = alg_x + BLK_W + MID_GAP
hw_box_w = BLK_W + 12
hw_box_h = HW_COL_H + 10
hw_box_x = hw_x - 6
hw_box_y = blk_y - 4
# "Hardware" label centered above the sub-box
lbl(root, hw_box_x, lbl_y, '<i>Hardware</i>', fs=7, fc='#888',
    al='center', w=hw_box_w, h=LBL_H)
# Dashed sub-box
cell(root, '', hw_box_x, hw_box_y, hw_box_w, hw_box_h,
     f'rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor={C_BD};'
     f'strokeWidth=0.6;dashed=1;dashPattern=4 2;')
# Per-channel LUT (top)
box(root, hw_x, blk_y, BLK_W, BLK_H,
    '<b>Per-channel LUT</b>', C_BRG, fs=9)
# FPGA Pipeline (bottom)
box(root, hw_x, blk_y + BLK_H + BLK_GAP, BLK_W, BLK_H,
    '<b>FPGA Pipeline</b>', C_HW, fs=9)

# ─── Arrow: Algorithm → Hardware ─────────────────────────────────
arr(root, alg_x + BLK_W, alg_blk_y + BLK_H // 2,
         hw_x, alg_blk_y + BLK_H // 2)

# ─── SOH ─────────────────────────────────────────────────────────
box(root, x_soh, soh_y, SM_W, SM_H, '<b>SOH</b>', C_IO, fs=9)

# ─── Main arrows ─────────────────────────────────────────────────
arr(root, x_train + BW, main_cy, x_co, main_cy)
arr(root, x_co + CO_W, main_cy, x_soh, main_cy)

# ═══════════════════════════════════════════════════════════════════
tree = ET.ElementTree(mxfile)
ET.indent(tree, space='  ')
out = '/home/orchid/Desktop/Project/PINN_FPGA/results/figures/src/fig_codesign_flow.drawio'
tree.write(out, xml_declaration=True, encoding='UTF-8')
print(f'Saved: {out}')
