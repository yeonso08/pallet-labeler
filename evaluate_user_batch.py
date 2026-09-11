"""Audit user-saved predictions against manual corrections without retraining."""
from pathlib import Path
import hashlib
import json
import re
import joblib
import numpy as np
from engine import read_cloud, truth_on_raw, infer, evaluate
from diagnose_v3 import errors

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/user_batch_400'

def inventory(folder):
    result={}
    for p in folder.glob('*.ply'):
        match=re.fullmatch(r'(\d+)(?:_labeled)*',p.stem)
        if not match:raise ValueError('Unrecognized filename: '+str(p))
        key=match[1]
        if key in result:raise ValueError('Ambiguous pair: '+key)
        result[key]=p
    return result

def main():
    desktop=Path('/Users/hwangjaeyeon/Desktop')
    raw,auto,corrected=[inventory(desktop/n) for n in ('01_original','02_v4_auto','03_corrected')]
    old=json.loads((ROOT/'results/training_v3/manifest.json').read_text())
    hashes={r['coordinate_sha256']:r['id'] for r in old['files']}
    v2=joblib.load(ROOT/'model.joblib');v2['default_split']=.06
    v4=joblib.load(ROOT/'results/segmentation_v4/model_v4.joblib')
    records=[];audit=[]
    for key in sorted(raw):
        ply,xyz=read_cloud(raw[key]);row=dict(id=key,raw=str(raw[key]),auto=str(auto.get(key,'')),corrected=str(corrected.get(key,'')),points=len(xyz),previous_exact_match=hashes.get(hashlib.sha256(xyz.tobytes()).hexdigest()))
        if key not in auto or key not in corrected:
            row['status']='missing_auto_or_correction';audit.append(row);continue
        truth,_=read_cloud(corrected[key]);a=truth['vertex'].data
        fields=[n for n in a.dtype.names if 'instance_label' in n]
        if fields!=['scalar_instance_label']:raise ValueError(f'{key}: ambiguous labels {fields}')
        labels=a['scalar_instance_label']
        if not np.all(np.isfinite(labels)&(labels>=1)&(labels==np.floor(labels))):raise ValueError(key+': invalid labels')
        y=truth_on_raw(ply,corrected[key]);saved=truth_on_raw(ply,auto[key])
        row.update(status='evaluated',truth_match_fraction=float(np.mean(y>0)),auto_match_fraction=float(np.mean(saved>0)))
        if min(row['truth_match_fraction'],row['auto_match_fraction'])<.95:raise ValueError(key+': inadequate coordinate coverage')
        pred4=infer(ply,v4)[0];pred2=infer(ply,v2)[0]
        good=(y>0)&(saved>0)
        # All three are scored on exactly the same matched points.
        row['scored_fraction']=float(np.mean(good))
        row['saved_vs_current_v4_label_agreement']=float(np.mean(saved[saved>0]==pred4[saved>0]))
        for name,p in [('saved_auto',saved),('v2',pred2),('v4',pred4)]:
            records.append(dict(id=key,model=name,**evaluate(y[good],p[good]),**errors(y[good],p[good])))
        audit.append(row)
        print('EVALUATED',key,'saved_v4_agreement',row['saved_vs_current_v4_label_agreement'],flush=True)
    summary={name:{k:float(np.mean([r[k] for r in records if r['model']==name])) for k in ('pallet_iou','object_mean_iou','object_f1_at_50','split_objects','merged_predictions','object_to_pallet','pallet_to_object')} for name in ('saved_auto','v2','v4')}
    result=dict(counts=dict(raw=len(raw),auto=len(auto),corrected=len(corrected)),audit=audit,summary=summary,records=records,note='Manual corrections treated as ground truth. No training or tuning on this batch.')
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
