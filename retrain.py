"""Reproducible training from an audited manifest; never modifies source clouds."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import joblib
from engine import read_cloud,rasterize,truth_on_raw,grid_truth,edge_features

def prepare(row,cache):
    digest=hashlib.sha256(b'pallet-features-v1-cell4-seed42')
    for key in ('raw','truth'):
        with Path(row[key]).open('rb') as source:
            for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    digest.update(row.get('label_source','scalar_instance_label').encode())
    dest=cache/(row['id']+'_'+digest.hexdigest()[:20]+'.npz')
    if dest.exists():return dest
    ply,_=read_cloud(row['raw']);r=rasterize(ply)
    y=truth_on_raw(ply,row['truth'],label_source=row.get('label_source','scalar_instance_label'))
    if np.mean(y>0)<.95:raise ValueError('Less than 95% exact-coordinate label matches: '+row['id'])
    g=grid_truth(r,y);rng=np.random.default_rng(int(row['id'].split('_')[-1])+42)
    ix=np.flatnonzero((g.ravel()>0)&r['valid'].ravel());ix=rng.choice(ix,min(14000,len(ix)),replace=False)
    x=r['features'].reshape(-1,r['features'].shape[-1])[ix];ys=(g.ravel()[ix]>1).astype(np.uint8)
    es=[];ets=[]
    for axis in (0,1):
        a=g[:,:-1] if axis==1 else g[:-1,:];b=g[:,1:] if axis==1 else g[1:,:]
        good=((a>0)&(b>0)&((a>1)|(b>1))).ravel();target=(a!=b).ravel()
        ef=edge_features(r,axis).reshape(-1,2*r['features'].shape[-1]);ids=[]
        for val,n in [(True,3500),(False,7000)]:
            pool=np.flatnonzero(good&(target==val));ids.extend(rng.choice(pool,min(n,len(pool)),replace=False))
        es.append(ef[ids]);ets.append(target[ids].astype(np.uint8))
    np.savez_compressed(dest,x=x,y=ys,e=np.concatenate(es),ey=np.concatenate(ets))
    return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());rows=[r for r in manifest['files'] if r['split']=='train'];args.cache.mkdir(parents=True,exist_ok=True)
    prepared=[]
    for i,row in enumerate(rows):
        prepared.append(prepare(row,args.cache));print('PREPARED',i+1,len(rows),row['id'],flush=True)
    if args.prepare_only:return
    if args.output is None:p.error('--output required')
    xs=[];ys=[];es=[];eys=[]
    for path in prepared:
        with np.load(path) as a:xs.append(a['x']);ys.append(a['y']);es.append(a['e']);eys.append(a['ey'])
    print('FITTING semantic',flush=True)
    sem=ExtraTreesClassifier(n_estimators=70,min_samples_leaf=3,max_depth=23,n_jobs=4,random_state=42)
    sem.fit(np.concatenate(xs),np.concatenate(ys));del xs,ys
    print('FITTING edge',flush=True)
    edge=ExtraTreesClassifier(n_estimators=70,min_samples_leaf=3,max_depth=23,n_jobs=4,random_state=43)
    edge.fit(np.concatenate(es),np.concatenate(eys));del es,eys
    model={'semantic':sem,'edge':edge,'cell':4.,'training_files':[r['id'] for r in rows],'format_version':1,'coordinate_system':'Original scanner XYZ; units as supplied; z is height','manifest':manifest,'default_split':.1,'default_min_cells':150}
    args.output.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,args.output,compress=3);print('SAVED',args.output,flush=True)
if __name__=='__main__':main()
