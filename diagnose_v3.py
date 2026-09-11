"""Read-only model ablations; outputs diagnostic metrics, not tuned models."""
import json
from pathlib import Path
import joblib
import numpy as np
from engine import read_cloud, truth_on_raw, predict_grid, segment_prediction, evaluate

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/training_v3'

def errors(y, p):
    valid=y>0; y=y[valid]; p=p[valid]
    a, ai=np.unique(y, return_inverse=True); b, bi=np.unique(p, return_inverse=True)
    table=np.bincount(ai*len(b)+bi, minlength=len(a)*len(b)).reshape(len(a),len(b))
    obj=table[np.ix_(a>1,b>1)]
    ts=table.sum(1)[a>1]; ps=table.sum(0)[b>1]
    # Ignore tiny overlaps: split fragments cover >=10% of a true object;
    # merge contributors cover >=10% of the predicted object, >=50 points.
    splits=((obj>=.1*ts[:,None]) & (obj>=50)).sum(1)
    merges=((obj>=.1*ps[None,:]) & (obj>=50)).sum(0)
    return dict(split_objects=int((splits>=2).sum()), merged_predictions=int((merges>=2).sum()),
                object_to_pallet=float(np.mean(p[y>1]==1)),
                pallet_to_object=float(np.mean(p[y==1]>1)),
                split_labels=a[a>1][splits>=2].tolist(),
                merge_labels=b[b>1][merges>=2].tolist())

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    models=[joblib.load(ROOT/'model.joblib'),joblib.load(OUT/'model_v3.joblib')]
    records=[]
    for row in manifest['files']:
        if row['split'] not in ('old_test','new_test','test'):continue
        ply,_=read_cloud(row['raw']); y=truth_on_raw(ply,row['truth'])
        pred=[predict_grid(ply,m) for m in models]
        for si,ei in ((0,0),(1,1),(0,1),(1,0)):
            for threshold in (.03,.06):
                labels,*_=segment_prediction(pred[si][0],pred[si][1],pred[ei][2],split=threshold)
                records.append(dict(file=Path(row['raw']).name,group=row['split'],
                    semantic='v'+str(si+2),edge='v'+str(ei+2),threshold=threshold,
                    **evaluate(y,labels),**errors(y,labels)))
        print('DIAGNOSED',Path(row['raw']).name,flush=True)
    groups={}
    for group in ('retained','fresh'):
        groups[group]={}
        for si,ei in ((2,2),(3,3),(2,3),(3,2)):
            for t in (.03,.06):
                rs=[r for r in records if (r['group']=='test')==(group=='fresh') and r['semantic']==f'v{si}' and r['edge']==f'v{ei}' and r['threshold']==t]
                keys=('object_mean_iou','object_f1_at_50','pallet_iou','split_objects','merged_predictions','object_to_pallet','pallet_to_object','predicted_objects','true_objects')
                groups[group][f's{si}_e{ei}_{t}']={k:float(np.mean([r[k] for r in rs])) for k in keys}
    result=dict(summary=groups,records=records,
                note='Diagnostic use only. Split/merge counts use 10% overlap and 50-point thresholds. No model or production setting changes.')
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(groups),flush=True)

if __name__=='__main__':main()
