"""Display-only point cloud helpers. No changes to coordinates or labels."""
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

BACKGROUND='#111b2b'
PALETTE=np.array([to_rgb(c) for c in ['#2374ef','#ffad24','#1cdbc8','#f75069','#a979ff','#d6ed40','#ff75d2','#30c3fa','#ff7744','#70e35b','#e9cc8b','#9fafff','#beecff','#b59054','#e85cff','#16b895','#f5e153','#baea88','#fbaecb','#838bff']],dtype=np.float32)

def display_indices(count,budget):
    if budget is None or count<=budget:return np.arange(count)
    # Regular subsampling exaggerates the scanner's scan-line pattern.
    return np.sort(np.random.default_rng(42).choice(count,budget,replace=False))

def point_colors(ply,xyz,labels,indices,mode,selected=None):
    if mode=='원본 색상':
        a=ply['vertex'].data
        if all(n in a.dtype.names for n in ('red','green','blue')):
            rgb=np.column_stack([a[n][indices] for n in ('red','green','blue')]).astype(np.float32)/255
        else:rgb=PALETTE[(labels[indices]-1)%len(PALETTE)].copy()
    elif mode=='높이 색상':
        lo,hi=np.quantile(xyz[:,2],[.01,.99]);v=np.clip((xyz[indices,2]-lo)/max(hi-lo,1e-8),0,1)
        rgb=colormaps['turbo'](v)[:,:3].copy()
    else:rgb=PALETTE[(labels[indices]-1)%len(PALETTE)].copy()
    if selected is not None:rgb[selected[indices]]=[1,1,1]
    return rgb
