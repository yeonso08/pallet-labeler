"""Plot corrected instances and predictions, and audit disconnected pieces."""
from pathlib import Path
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from engine import read_cloud,truth_on_raw,predict_grid,segment_prediction,grid_truth

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/corrected7_v5'

def main():
    rows=json.loads((ROOT/'results/user_batch_400/evaluation.json').read_text())['audit']
    model=joblib.load(ROOT/'results/segmentation_v4/model_v4.joblib')
    records=[]
    for row in rows:
        if row['id'] not in ('401','406'):continue
        ply,_=read_cloud(row['raw']);y=truth_on_raw(ply,row['corrected'])
        r,prob,boundary=predict_grid(ply,model)
        p,_,_,pg,_=segment_prediction(r,prob,boundary,split=.03,hole_fill_probability=.5)
        gt=grid_truth(r,y)
        fig,axes=plt.subplots(1,2,figsize=(14,7))
        for ax,grid,title in zip(axes,(gt,pg),('Manual correction','v4 automatic')):
            ax.imshow(np.ma.masked_where((grid<=1)|(~r['valid']),grid),origin='lower',cmap='tab20',vmin=2,vmax=21,interpolation='nearest')
            for lab in np.unique(grid[grid>1]):
                yy,xx=np.where((grid==lab)&r['valid'])
                if len(xx):ax.text(np.mean(xx),np.mean(yy),str(lab),fontsize=10,bbox=dict(facecolor='white',alpha=.8,pad=1))
            ax.set_title(row['id']+' - '+title);ax.set_xlabel('X grid (4 units/cell)');ax.set_ylabel('Y grid')
        fig.tight_layout();fig.savefig(OUT/(row['id']+'_instances.png'),dpi=140);plt.close(fig)
        cc,n=ndi.label(pg>1)
        for label in np.unique(y[y>1]):
            mask=y==label;pred_ids,counts=np.unique(p[mask],return_counts=True)
            dominant=[int(a) for a,c in zip(pred_ids,counts) if a>1 and c>=.1*mask.sum()]
            if len(dominant)<2:continue
            components=np.unique(cc[np.isin(pg,dominant)]);components=components[components>0]
            records.append(dict(file=row['id'],truth_label=int(label),predicted_pieces=dominant,object_mask_components=components.tolist()))
        print('PLOTTED',row['id'],flush=True)
    (OUT/'geometry.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records),flush=True)

if __name__=='__main__':main()
