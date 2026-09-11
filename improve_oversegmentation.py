"""Compare seed filtering or smoothing on validation; retain baseline if no gain."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
from engine import read_cloud,truth_on_raw,predict_grid,segment_prediction,evaluate

ROOT=Path(__file__).resolve().parent
KEYS=('pallet_iou','object_mean_iou','object_f1_at_50')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('size','smoothing'),default='smoothing');args=parser.parse_args()
    OUT=ROOT/('results/segmentation_v5' if args.mode=='size' else 'results/segmentation_v5_smoothing')
    baseline_parameter=150 if args.mode=='size' else 1
    parameter_name='min_seed_cells' if args.mode=='size' else 'seed_smoothing'
    OUT.mkdir(parents=True,exist_ok=True)
    model=joblib.load(ROOT/'results/segmentation_v4/model_v4.joblib')
    rows=json.loads((ROOT/'results/training_v3/manifest.json').read_text())['files']
    batch=json.loads((ROOT/'results/user_batch_400/evaluation.json').read_text())['audit']
    configs=[(s,n) for s in (.03,.06,.1) for n in ((150,300,600) if args.mode=='size' else (1,3,5))]
    results=[]
    for row in rows:
        if row['split']!='validation':continue
        ply,_=read_cloud(row['raw']);y=truth_on_raw(ply,row['truth']);pred=predict_grid(ply,model)
        for s,n in configs:
            p,*_=segment_prediction(*pred,split=s,hole_fill_probability=.5,**{parameter_name:n})
            results.append(dict(id=row['id'],split=s,seed=n,**evaluate(y,p)))
        print('VALIDATED',row['id'],flush=True)
    summary=[dict(split=s,seed=n,**{k:float(np.mean([r[k] for r in results if r['split']==s and r['seed']==n])) for k in KEYS}) for s,n in configs]
    base=next(r for r in summary if r['split']==.03 and r['seed']==baseline_parameter)
    eligible=[r for r in summary if all(r[k]>=base[k]-1e-12 for k in KEYS)]
    best=max(eligible,key=lambda r:(r['object_f1_at_50'],r['object_mean_iou']))
    (OUT/'validation.json').write_text(json.dumps(dict(summary=summary,selected=best,records=results,parameter_name=parameter_name),indent=2))
    print('SELECTED',best,flush=True)
    checks=[]
    checkrows=[dict(id=r['id'],raw=r['raw'],truth=r['corrected'],group='corrected7') for r in batch if r['status']=='evaluated']
    checkrows += [dict(r,group=r['split']) for r in rows if r['split'] in ('old_test','new_test','test')]
    for row in checkrows:
        ply,_=read_cloud(row['raw']);y=truth_on_raw(ply,row['truth']);pred=predict_grid(ply,model)
        for name,s,n in [('v4',.03,baseline_parameter),('candidate',best['split'],best['seed'])]:
            p,*_=segment_prediction(*pred,split=s,hole_fill_probability=.5,**{parameter_name:n})
            checks.append(dict(id=row['id'],group=row['group'],model=name,**evaluate(y,p)))
        print('CHECKED',row['id'],flush=True)
    sums={g:{m:{k:float(np.mean([r[k] for r in checks if r['group']==g and r['model']==m])) for k in KEYS} for m in ('v4','candidate')} for g in ('corrected7','old_test','new_test','test')}
    (OUT/'regression.json').write_text(json.dumps(dict(summary=sums,records=checks),indent=2))
    model.update(default_split=best['split'],segmentation_version=5,**{parameter_name:best['seed']})
    joblib.dump(model,OUT/'model_v5.joblib',compress=3)
    print('RESULT',json.dumps(sums),flush=True)

if __name__=='__main__':main()
