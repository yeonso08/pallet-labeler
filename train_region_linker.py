"""Train occluded-piece affinity and select linking confidence on validation only."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from engine import read_cloud,truth_on_raw,infer,evaluate
from region_affinity import region_pairs,merge_regions

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/region_link_v5'
KEYS=('pallet_iou','object_mean_iou','object_f1_at_50')

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'results/corrected7_v5/manifest.json').read_text())
    base=joblib.load(ROOT/'results/segmentation_v4/model_v4.joblib')
    xs=[];ys=[];training=[]
    for row in manifest['files']:
        if row['split']!='train':continue
        ply,_=read_cloud(row['raw']);truth=truth_on_raw(ply,row['truth'])
        labels,_,r,grid,_=infer(ply,base);ids,pairs,x=region_pairs(r,grid)
        dominant={}
        for i,lab in enumerate(ids):
            vals,counts=np.unique(truth[(labels==lab)&(truth>0)],return_counts=True)
            if len(vals) and vals[counts.argmax()]>1 and counts.max()/counts.sum()>=.9:
                dominant[i]=int(vals[counts.argmax()])
        valid=np.array([i in dominant and j in dominant for i,j in pairs],dtype=bool)
        if valid.any():
            target=np.array([int(dominant[i]==dominant[j]) for i,j in pairs[valid]],dtype=np.uint8)
            xs.append(x[valid]);ys.append(target)
            training.append(dict(id=row['id'],pairs=len(target),positive=int(target.sum())))
        print('PREPARED LINK',row['id'],flush=True)
    x=np.concatenate(xs);y=np.concatenate(ys)
    if len(np.unique(y))!=2:raise ValueError('Need positive and negative link examples')
    clf=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,max_depth=16,n_jobs=4,random_state=51,class_weight='balanced')
    clf.fit(x,y);joblib.dump(clf,OUT/'linker.joblib',compress=3)
    save('training.json',dict(scenes=training,pairs=len(y),positive=int(y.sum()),features=x.shape[1]))
    thresholds=(1.,.98,.95,.9,.8,.7,.6)
    validation=[]
    for row in manifest['files']:
        if row['split']!='validation':continue
        ply,_=read_cloud(row['raw']);truth=truth_on_raw(ply,row['truth'])
        _,_,r,grid,_=infer(ply,base);ids,pairs,x=region_pairs(r,grid)
        scores=clf.predict_proba(x)[:,1] if len(pairs) else []
        for t in thresholds:
            g=merge_regions(grid,ids,pairs,scores,t)
            validation.append(dict(id=row['id'],threshold=t,**evaluate(truth,g.ravel()[r['flat']])))
        print('VALIDATED LINK',row['id'],flush=True)
    sums=[dict(threshold=t,**{k:float(np.mean([r[k] for r in validation if r['threshold']==t])) for k in KEYS}) for t in thresholds]
    baseline=sums[0]
    eligible=[r for r in sums if all(r[k]>=baseline[k]-1e-10 for k in KEYS)]
    best=max(eligible,key=lambda r:(r['object_f1_at_50'],r['object_mean_iou'],r['threshold']))
    save('validation.json',dict(summary=sums,selected=best,records=validation))
    print('SELECTED LINK',best,flush=True)
    candidate=dict(base,region_linker=clf,region_link_threshold=best['threshold'],model_version='region_link_v5',
                   region_link_training_files=[r['id'] for r in manifest['files'] if r['split']=='train'])
    joblib.dump(candidate,OUT/'model_v5.joblib',compress=3)
    records=[]
    for row in manifest['files']:
        group='training7' if row['id'].startswith('corrected_') else row['split']
        if group not in ('training7','old_test','new_test','test'):continue
        ply,_=read_cloud(row['raw']);truth=truth_on_raw(ply,row['truth'])
        _,_,r,grid,_=infer(ply,base);ids,pairs,x=region_pairs(r,grid)
        scores=clf.predict_proba(x)[:,1] if len(pairs) else []
        for name,t in [('v4',1.),('v5',best['threshold'])]:
            g=merge_regions(grid,ids,pairs,scores,t)
            records.append(dict(id=row['id'],group=group,model=name,**evaluate(truth,g.ravel()[r['flat']])))
        print('CHECKED LINK',row['id'],flush=True)
    summary={g:{m:{k:float(np.mean([r[k] for r in records if r['group']==g and r['model']==m])) for k in KEYS} for m in ('v4','v5')} for g in ('old_test','new_test','test','training7')}
    save('regression.json',dict(summary=summary,records=records))
    print('LINK COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
