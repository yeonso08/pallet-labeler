"""Select pallet-preserving hole filling on validation; report regression scores."""
import json
from pathlib import Path
import joblib
import numpy as np
from engine import read_cloud, truth_on_raw, predict_grid, segment_prediction, evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/segmentation_v4'
METRICS=('pallet_iou','object_mean_iou','object_f1_at_50')

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'results/training_v3/manifest.json').read_text())['files']
    models={'v2':joblib.load(ROOT/'model.joblib'), 'v3':joblib.load(ROOT/'results/training_v3/model_v3.joblib')}
    configs=[dict(model=m,split=s,hole_fill_probability=h) for m in models for s in (.03,.06) for h in (0.,.2,.5)]
    records=[]
    for row in rows:
        if row['split']!='validation':continue
        ply,_=read_cloud(row['raw']);y=truth_on_raw(ply,row['truth'])
        for name,model in models.items():
            pred=predict_grid(ply,model)
            for i,c in enumerate(configs):
                if c['model']!=name:continue
                p,*_=segment_prediction(*pred,split=c['split'],hole_fill_probability=c['hole_fill_probability'])
                records.append(dict(id=row['id'],config=i,**evaluate(y,p)))
        print('VALIDATION',row['id'],flush=True)
    summary=[dict(config=i,**c,**{k:float(np.mean([r[k] for r in records if r['config']==i])) for k in METRICS}) for i,c in enumerate(configs)]
    # Predeclared choice: require all three validation averages to meet v2
    # production baseline; among eligible pallet-preserving variants maximize IoU.
    base=next(r for r in summary if r['model']=='v2' and r['split']==.06 and r['hole_fill_probability']==0.)
    eligible=[r for r in summary if r['hole_fill_probability']>0 and all(r[k]>=base[k] for k in METRICS)]
    selected=max(eligible,key=lambda r:r['object_mean_iou']) if eligible else None
    save('validation.json',dict(summary=summary,selected=selected,records=records))
    print('SELECTED',json.dumps(selected),flush=True)
    if selected is None:return
    candidate=dict(models[selected['model']])
    candidate.update(default_split=selected['split'],hole_fill_probability=selected['hole_fill_probability'],segmentation_version=4)
    joblib.dump(candidate,OUT/'model_v4.joblib',compress=3)
    results=[]
    for row in rows:
        if row['split'] not in ('old_test','new_test','test'):continue
        ply,_=read_cloud(row['raw']);y=truth_on_raw(ply,row['truth'])
        for name,model,split,hole in [('v2',models['v2'],.06,0.),('v3',models['v3'],.03,0.),('v4',candidate,candidate['default_split'],candidate['hole_fill_probability'])]:
            p,*_=segment_prediction(*predict_grid(ply,model),split=split,hole_fill_probability=hole)
            results.append(dict(id=row['id'],group=row['split'],model=name,**evaluate(y,p)))
        print('REGRESSION',row['id'],flush=True)
    sums={g:{m:{k:float(np.mean([r[k] for r in results if r['group']==g and r['model']==m])) for k in METRICS} for m in ('v2','v3','v4')} for g in ('old_test','new_test','test')}
    save('regression.json',dict(summary=sums,records=results,note='Previously inspected scenes: regression evidence, not an untouched final test.'))
    print('RESULT',json.dumps(sums),flush=True)

if __name__=='__main__':main()
