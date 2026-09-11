"""Train with seven completed corrections; preserve v4 and all held-out splits."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import joblib
import numpy as np
from engine import read_cloud, truth_on_raw, infer, evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/corrected7_v5'
KEYS=('pallet_iou','object_mean_iou','object_f1_at_50')

def write(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False))

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.evaluate_only:
        manifest=json.loads((ROOT/'results/training_v3/manifest.json').read_text())
        audit=json.loads((ROOT/'results/user_batch_400/evaluation.json').read_text())['audit']
        added=[]
        for item in audit:
            if item['status']!='evaluated':continue
            raw,_=read_cloud(item['raw']);truth,_=read_cloud(item['corrected'])
            labels=truth['vertex'].data['scalar_instance_label']
            if not np.all(np.isfinite(labels)&(labels>=1)&(labels==np.floor(labels))):
                raise ValueError('Invalid labels '+item['id'])
            y=truth_on_raw(raw,item['corrected'])
            if np.mean(y>0)<.95:raise ValueError('Insufficient coverage '+item['id'])
            added.append(dict(id='corrected_'+item['id'],raw=item['raw'],truth=item['corrected'],
                              label_source='scalar_instance_label',split='train',
                              match_fraction=float(np.mean(y>0)),raw_sha256=digest(item['raw']),truth_sha256=digest(item['corrected'])))
        if len(added)!=7:raise ValueError('Expected exactly the seven completed corrections')
        manifest['files']+=added
        manifest['note']='128 previous training scenes + seven completed manual corrections (400–406). Previous 20 validation and 22 regression files retained. 407–409 excluded. New seven are now training data, not independent test data.'
        write('manifest.json',manifest)
        subprocess.run([sys.executable,str(ROOT/'retrain.py'),str(OUT/'manifest.json'),
                        '--cache',str(ROOT/'results/training_v3/cache'),'--output',str(OUT/'model_v5.joblib')],check=True)
        candidate=joblib.load(OUT/'model_v5.joblib')
        candidate.update(default_split=.03,hole_fill_probability=.5,default_min_cells=150,
                         model_version='corrected7_v5',segmentation_version=4)
        joblib.dump(candidate,OUT/'model_v5.joblib',compress=3)
    manifest=json.loads((OUT/'manifest.json').read_text())
    models={'v4':joblib.load(ROOT/'results/segmentation_v4/model_v4.joblib'),
            'v5':joblib.load(OUT/'model_v5.joblib')}
    results=[]
    for row in manifest['files']:
        if row['split']=='train' and not row['id'].startswith('corrected_'):continue
        group='training7' if row['split']=='train' else row['split']
        if group=='training7' and (digest(row['raw'])!=row['raw_sha256'] or digest(row['truth'])!=row['truth_sha256']):
            raise ValueError('Source changed during training: '+row['id'])
        ply,_=read_cloud(row['raw']);y=truth_on_raw(ply,row['truth'])
        for name,model in models.items():
            p,*_=infer(ply,model)
            results.append(dict(id=row['id'],group=group,model=name,**evaluate(y,p)))
        write('evaluation_progress.json',results)
        print('EVALUATED',group,row['id'],flush=True)
    summary={g:{m:{k:float(np.mean([r[k] for r in results if r['group']==g and r['model']==m])) for k in KEYS} for m in models} for g in ('validation','old_test','new_test','test','training7')}
    write('evaluation.json',dict(summary=summary,records=results,model_sha256=digest(OUT/'model_v5.joblib'),
                                note='Fixed v4 postprocessing. Training7 scores are in-sample; earlier inspected evaluation groups are regression evidence. No 407–409 corrections used.'))
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
