"""Scene-disjoint training, validation and held-out evaluation."""
from pathlib import Path
import argparse,json,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import joblib
from engine import read_cloud,rasterize,truth_on_raw,grid_truth,edge_features,infer,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent);args=p.parse_args()
 rng=np.random.default_rng(42);xs=[];ys=[];es=[];ets=[];used=[];skipped=[]
 for path in sorted(args.raw.glob('*.ply')):
  if not path.stem.isdigit() or int(path.stem)>=124:continue
  try:
   ply,_=read_cloud(path);r=rasterize(ply);y=truth_on_raw(ply,args.truth/path.name);g=grid_truth(r,y)
  except (ValueError,FileNotFoundError) as e:skipped.append({'file':path.name,'reason':str(e)});continue
  ix=np.flatnonzero((g.ravel()>0)&r['valid'].ravel());ix=rng.choice(ix,min(14000,len(ix)),replace=False)
  xs.append(r['features'].reshape(-1,r['features'].shape[-1])[ix]);ys.append((g.ravel()[ix]>1).astype(int))
  for axis in (0,1):
   a=g[:,:-1] if axis==1 else g[:-1,:];b=g[:,1:] if axis==1 else g[1:,:]
   good=(a>0)&(b>0)&((a>1)|(b>1));target=(a!=b)
   ef=edge_features(r,axis).reshape(-1,2*r['features'].shape[-1]);target=target.ravel();good=good.ravel()
   ids=[]
   for val,n in [(True,3500),(False,7000)]:
    pool=np.flatnonzero(good&(target==val));ids.extend(rng.choice(pool,min(n,len(pool)),replace=False))
   es.append(ef[ids]);ets.append(target[ids].astype(int))
  used.append(path.name);print('Prepared',path.name,'matched',round(np.mean(y>0),5),flush=True)
 print('Training classifiers',flush=True)
 sem=ExtraTreesClassifier(n_estimators=70,min_samples_leaf=3,max_depth=23,n_jobs=4,random_state=42)
 edge=ExtraTreesClassifier(n_estimators=70,min_samples_leaf=3,max_depth=23,n_jobs=4,random_state=43)
 sem.fit(np.concatenate(xs),np.concatenate(ys));edge.fit(np.concatenate(es),np.concatenate(ets))
 model={'semantic':sem,'edge':edge,'cell':4.,'training_files':used,'skipped':skipped,'format_version':1,'coordinate_system':'Original scanner XYZ; units as supplied; z is height'}
 args.output.mkdir(parents=True,exist_ok=True);joblib.dump(model,args.output/'model.joblib',compress=3)
 print('Model saved',flush=True)
 records=[]
 for name in range(124,134):
  path=args.raw/f'{name}.ply';ply,_=read_cloud(path);y=truth_on_raw(ply,args.truth/path.name)
  start=time.monotonic();labels,*_=infer(ply,model);m=evaluate(y,labels);m.update(file=path.name,split='validation' if name<128 else 'test',seconds=round(time.monotonic()-start,2));records.append(m);print(m,flush=True)
 (args.output/'evaluation.json').write_text(json.dumps({'training':used,'excluded':skipped,'results':records},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
