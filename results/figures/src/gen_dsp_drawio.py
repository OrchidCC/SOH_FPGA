#!/usr/bin/env python3
"""DSP packing — single-column, strict grid alignment, no overflow."""

import xml.etree.ElementTree as ET

CELL_ID=[2]
C_W1='#FFDCA8';C_W2='#B8E0B8';C_W3='#A8C8F0';C_ACT='#D0B8E0'
C_GRD='#F0F0F0';C_BDR='#333333'

def nid(): CELL_ID[0]+=1; return str(CELL_ID[0])

def cell(r,v,x,y,w,h,s):
    c=ET.SubElement(r,'mxCell',id=nid(),value=v,style=s,vertex='1',parent='1')
    ET.SubElement(c,'mxGeometry',x=str(int(x)),y=str(int(y)),width=str(int(w)),height=str(int(h)),**{'as':'geometry'})

def box(r,x,y,l,f,w=28,h=24,fs=9):
    cell(r,l,x,y,w,h,f'rounded=0;whiteSpace=wrap;html=1;fillColor={f};strokeColor={C_BDR};strokeWidth=0.6;fontSize={fs};fontFamily=Times New Roman;align=center;verticalAlign=middle;')

def txt(r,x,y,l,fs=9,c='#333',al='left',w=80,h=16,b=0):
    cell(r,l,x,y,w,h,f'text;html=1;fontSize={fs};fontFamily=Times New Roman;fontStyle={b};fontColor={c};align={al};verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;')

def arr(r,x1,y1,x2,y2):
    cid=nid();c=ET.SubElement(r,'mxCell',id=cid,value='',style='endArrow=classic;html=1;strokeColor=#555;strokeWidth=1;endSize=4;',edge='1',parent='1')
    g=ET.SubElement(c,'mxGeometry',relative='1',**{'as':'geometry'})
    ET.SubElement(g,'mxPoint',x=str(int(x1)),y=str(int(y1)),**{'as':'sourcePoint'})
    ET.SubElement(g,'mxPoint',x=str(int(x2)),y=str(int(y2)),**{'as':'targetPoint'})

def wblock(r,x,y,label,color,w=100,h=24,fs=9):
    """Wide labeled block (for sub-products)."""
    cell(r,label,x,y,w,h,f'rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={C_BDR};strokeWidth=0.6;fontSize={fs};fontFamily=Times New Roman;align=center;verticalAlign=middle;')

mxfile=ET.Element('mxfile')
diag=ET.SubElement(mxfile,'diagram',name='DSP Packing')
mdl=ET.SubElement(diag,'mxGraphModel',dx='0',dy='0',grid='1',gridSize='4',guides='1')
root=ET.SubElement(mdl,'root')
ET.SubElement(root,'mxCell',id='0')
ET.SubElement(root,'mxCell',id='1',parent='0')

# ─── Layout constants ─────────────────────────────────────────────
# Total width target: ~400px (fits single column at reasonable DPI)
CW=28; CH=22
COL1=0    # section label
COL2=70   # data start
COL3=200  # secondary info
RH=26     # row height

Y=0

# ═══════════════════════════════════════════════════════════════════
# (a) A-port packing — single 25-bit horizontal register
# ═══════════════════════════════════════════════════════════════════
txt(root,COL1,Y,'<b>(a) Weight packing</b>',fs=10,w=160)
Y+=22

# Proportional widths: ~12px per bit
BW1=24   # 3b unused
BW4=40   # 4b weight
BW5=36   # 5b guard
REG_H=24
C_GUARD='#F0F0F0'
C_UNUSED='#F0F0F0'

# Register segments left-to-right: [24:22] [21:18] [17:13] [12:9] [8:4] [3:0]
rx=COL1+4
segs = [
    (BW1, '0',   C_UNUSED, '[24:22]', ''),
    (BW4, 'w<sub><i>j</i>+2</sub>', C_W3,    '[21:18]', ''),
    (BW5, '0', C_GUARD, '[17:13]', ''),
    (BW4, 'w<sub><i>j</i>+1</sub>', C_W2,    '[12:9]',  ''),
    (BW5, '0', C_GUARD, '[8:4]',   ''),
    (BW4, 'w<sub><i>j</i></sub>', C_W1,    '[3:0]',   ''),
]

# Bit labels above, register boxes, neuron labels below
for (sw, label, color, bits, neuron) in segs:
    # bit range above
    txt(root, rx, Y, bits, fs=7, c='#555', al='center', w=sw, h=12)
    # register cell
    cell(root, label, rx, Y+14, sw, REG_H,
         f'rounded=0;whiteSpace=wrap;html=1;fillColor={color};strokeColor={C_BDR};'
         f'strokeWidth=0.6;fontSize=8;fontFamily=Times New Roman;align=center;verticalAlign=middle;')
    # neuron label below (only for weight segments)
    if neuron:
        txt(root, rx, Y+14+REG_H+1, neuron, fs=7, c='#444', al='center', w=sw, h=12)
    rx += sw

# B-port label on the right
txt(root, rx+6, Y+14+6, '× <i>x</i>', fs=9, c='#333', al='left', w=24, h=16)

# Column references for sections (c) and (d)
C_LBL=COL1+4
C_BIT=COL1+44
C_CEL=COL1+96
C_GI=C_CEL+4*CW+8

# Advance Y past register + neuron labels
Y += 14 + REG_H + 12

# ═══════════════════════════════════════════════════════════════════
# (b) Product extraction
# ═══════════════════════════════════════════════════════════════════
Y+=10
txt(root,COL1,Y,'<b>(b) Sub-products</b>',fs=10,w=160)
Y+=18

# Reuse same column grid
P_LBL=C_LBL
P_DESC=C_BIT
P_BLK=C_CEL
P_BITS=C_GI

BH=20  # compact block height

# Neuron j
txt(root,P_LBL,Y+2,'Neuron <i>j</i>',fs=8,c='#A05020',w=64)
wblock(root,P_BLK,Y,'w<sub><i>j</i></sub> × <i>x</i>',C_W1,w=110,h=BH,fs=8)
txt(root,P_BITS,Y+2,'[7:0]',fs=8,c='#555',w=40)
Y+=RH

# Neuron j+1
txt(root,P_LBL,Y+2,'Neuron <i>j</i>+1',fs=8,c='#207020',w=64)
wblock(root,P_BLK,Y,'w<sub><i>j</i>+1</sub> × <i>x</i>',C_W2,w=110,h=BH,fs=8)
txt(root,P_BITS,Y+2,'[16:9]',fs=8,c='#555',w=40)
Y+=RH

# Neuron j+2
txt(root,P_LBL,Y+2,'Neuron <i>j</i>+2',fs=8,c='#2060A0',w=64)
wblock(root,P_BLK,Y,'w<sub><i>j</i>+2</sub> × <i>x</i>',C_W3,w=110,h=BH,fs=8)
txt(root,P_BITS,Y+2,'[25:18]',fs=8,c='#555',w=40)



# ═══════════════════════════════════════════════════════════════════
tree=ET.ElementTree(mxfile);ET.indent(tree,space='  ')
out='/home/orchid/Desktop/Project/PINN_FPGA/results/figures/src/fig_dsp_packing.drawio'
tree.write(out,xml_declaration=True,encoding='UTF-8')
print(f'Saved: {out}')
