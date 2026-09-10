"""Local point-cloud instance labeling. No cloud services or source-file writes."""
from pathlib import Path
import json
import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed
import joblib

LABEL='scalar_instance_label'

def read_cloud(path):
    ply=PlyData.read(str(path)); a=ply['vertex'].data
    if not all(k in a.dtype.names for k in ('x','y','z')): raise ValueError('x, y, z 좌표가 없는 PLY입니다.')
    xyz=np.column_stack([a[k] for k in ('x','y','z')]).astype(np.float32)
    if len(xyz)==0 or not np.isfinite(xyz).all(): raise ValueError('빈 파일 또는 유효하지 않은 좌표가 있습니다.')
    return ply,xyz

def save_cloud(ply, labels, path, review=None):
    path=Path(path)
    if path.exists(): raise FileExistsError(f'이미 존재하는 파일입니다: {path}')
    a=ply['vertex'].data
    names=[n for n in a.dtype.names if n not in (LABEL,'instance_label','scalar_review_needed')]
    dtype=[(n,a.dtype.fields[n][0]) for n in names]+[(LABEL,'<f4')]
    if review is not None: dtype += [('scalar_review_needed','<f4')]
    out=np.empty(len(a),dtype=dtype)
    for n in names: out[n]=a[n]
    out[LABEL]=labels
    if review is not None:out['scalar_review_needed']=review
    elements=[PlyElement.describe(out,'vertex') if e.name=='vertex' else e for e in ply.elements]
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation protects previously saved results as well as originals.
    with path.open('xb') as f:
        PlyData(elements,text=False,byte_order='<',comments=list(ply.comments)+['Auto labels: manually review before training']).write(f)

def rasterize(ply, cell=4.):
    a=ply['vertex'].data; xyz=np.column_stack([a[k] for k in ('x','y','z')]).astype(np.float32)
    lo=xyz[:,:2].min(0); ij=np.floor((xyz[:,:2]-lo)/cell).astype(int)
    w,h=ij.max(0)+1
    if w*h>2_000_000: raise ValueError('좌표 범위가 너무 큽니다. 이 모델은 제공된 스캐너 좌표/단위를 기준으로 합니다.')
    flat=ij[:,1]*w+ij[:,0]; count=np.bincount(flat,minlength=w*h); valid=count.reshape(h,w)>0
    chans=[xyz[:,2]]
    for k in ('red','green','blue','nx','ny','nz'):
        chans.append(a[k].astype(np.float32) if k in a.dtype.names else np.zeros(len(a),np.float32))
    means=np.stack([np.bincount(flat,weights=c,minlength=w*h)/np.maximum(count,1) for c in chans],axis=1).reshape(h,w,-1).astype(np.float32)
    nearest=ndi.distance_transform_edt(~valid,return_distances=False,return_indices=True)
    means[~valid]=means[tuple(nearest[:,~valid])]
    z=means[:,:,0]; sm=ndi.median_filter(z,size=3)
    gx=ndi.sobel(sm,axis=1)/8;gy=ndi.sobel(sm,axis=0)/8
    yy,xx=np.indices((h,w));coords=np.stack([lo[0]+xx*cell,lo[1]+yy*cell],axis=-1)
    feats=[means,coords, gx[:,:,None],gy[:,:,None],(z-sm)[:,:,None]]
    for size in (3,7,15):
        mn=ndi.minimum_filter(sm,size=size);mx=ndi.maximum_filter(sm,size=size);avg=ndi.uniform_filter(sm,size=size)
        feats += [(z-mn)[:,:,None],(mx-z)[:,:,None],(z-avg)[:,:,None]]
    features=np.concatenate(feats,axis=2).astype(np.float32)
    return dict(features=features,means=means,valid=valid,flat=flat,shape=(h,w),lo=lo,cell=cell,xyz=xyz)

def truth_on_raw(raw_ply, truth_path, label_source=LABEL):
    truth, xyz=read_cloud(truth_path); names=truth['vertex'].data.dtype.names
    if label_source not in (LABEL,'scalar_Original_cloud_index'):raise ValueError('Unsupported label source')
    if label_source not in names:raise ValueError('정답 라벨이 없습니다.')
    if label_source==LABEL and len([n for n in names if 'instance_label' in n])!=1: raise ValueError('중복 정답 라벨 필드가 있습니다.')
    raw=np.column_stack([raw_ply['vertex'].data[k] for k in ('x','y','z')]);d,ix=cKDTree(xyz).query(raw,workers=4)
    y=truth['vertex'].data[label_source][ix].copy()
    if label_source=='scalar_Original_cloud_index': y=y+1
    good=(d<1e-4)&np.isfinite(y)&(y>=1)&(y==np.floor(y))
    result=np.full(len(raw),-1,np.int32);result[good]=y[good].astype(np.int32)
    return result

def grid_truth(r, point_labels):
    flat=r['flat'];size=np.prod(r['shape']);classes=np.unique(point_labels[point_labels>0])
    counts=np.stack([np.bincount(flat[point_labels==c],minlength=size) for c in classes])
    win=counts.argmax(0);total=counts.sum(0);mx=counts.max(0)
    result=classes[win]; result[(total==0)|(mx<.8*total)]=-1
    return result.reshape(r['shape'])

def edge_features(r,axis):
    f=r['features'];a=f[:,:-1] if axis==1 else f[:-1,:];b=f[:,1:] if axis==1 else f[1:,:]
    return np.concatenate([np.abs(a-b),(a+b)*.5],axis=-1).astype(np.float32)

def predict_grid(ply,model):
    r=rasterize(ply,model['cell']);f=r['features'];h,w=r['shape']
    prob=model['semantic'].predict_proba(f.reshape(-1,f.shape[-1]))[:,1].reshape(h,w)
    boundary=np.zeros((h,w),np.float32)
    for axis in (0,1):
        ef=edge_features(r,axis);p=model['edge'].predict_proba(ef.reshape(-1,ef.shape[-1]))[:,1].reshape(ef.shape[:2])
        if axis==1:
            boundary[:,:-1]=np.maximum(boundary[:,:-1],p);boundary[:,1:]=np.maximum(boundary[:,1:],p)
        else:
            boundary[:-1,:]=np.maximum(boundary[:-1,:],p);boundary[1:,:]=np.maximum(boundary[1:,:],p)
    return r,prob,boundary

def segment_prediction(r,prob,boundary,split=.1,min_cells=150):
    h,w=r['shape']
    obj=prob>.5
    # Bridge only small scanner sampling holes, never the outer empty area.
    support=ndi.distance_transform_edt(~r['valid'])<=1.5
    obj &= support
    obj=ndi.binary_fill_holes(obj)&support
    safe=obj&(boundary<split)
    markers,n=ndi.label(safe)
    sizes=np.bincount(markers.ravel());keep=sizes>=min_cells;keep[0]=False
    markers[~keep[markers]]=0;markers,_=ndi.label(markers>0)
    # Every isolated object component needs a seed, including uncertain small pieces.
    cc,n=ndi.label(obj);next_id=int(markers.max())
    for c in range(1,n+1):
        mask=cc==c
        if mask.sum()<min_cells:obj[mask]=False;continue
        if not np.any(markers[mask]):
            candidates=np.flatnonzero(mask);q=candidates[np.argmin(boundary.ravel()[candidates])];next_id+=1;markers.ravel()[q]=next_id
    seg=watershed(boundary,markers=markers,mask=obj)
    ids,cnts=np.unique(seg[seg>0],return_counts=True)
    labels_grid=np.ones((h,w),np.int32)
    for i,lab in enumerate(ids,2): labels_grid[seg==lab]=i
    labels=labels_grid.ravel()[r['flat']]
    uncertain=((prob>.2)&(prob<.8))|(boundary>.35)
    review=uncertain.ravel()[r['flat']].astype(np.float32)
    return labels,review,r,labels_grid,dict(objects=len(ids),review_fraction=float(review.mean()),split=split,min_cells=min_cells)

def infer(ply,model,split=None,min_cells=None):
    if split is None:split=model.get('default_split',.1)
    if min_cells is None:min_cells=model.get('default_min_cells',150)
    return segment_prediction(*predict_grid(ply,model),split=split,min_cells=min_cells)

def evaluate(y,p):
    keep=y>0;y=y[keep];p=p[keep]
    if not len(y):raise ValueError('No matched ground-truth points to evaluate')
    yl,yi=np.unique(y,return_inverse=True);pl,pi=np.unique(p,return_inverse=True)
    counts=np.bincount(yi*len(pl)+pi,minlength=len(yl)*len(pl)).reshape(len(yl),len(pl))
    true=yl>1;pred=pl>1
    inter=counts[np.ix_(true,pred)]
    union=counts.sum(1)[true,None]+counts.sum(0)[None,pred]-inter
    mat=inter/np.maximum(union,1)
    ri,ci=linear_sum_assignment(-mat) if mat.size else ([],[])
    ious=mat[ri,ci] if mat.size else np.array([])
    tp=int(np.sum(ious>=.5));pal=np.sum((y==1)&(p==1))/max(np.sum((y==1)|(p==1)),1)
    return dict(pallet_iou=float(pal),object_mean_iou=float(ious.sum()/max(true.sum(),1)),object_f1_at_50=float(2*tp/max(true.sum()+pred.sum(),1)),true_objects=int(true.sum()),predicted_objects=int(pred.sum()),scored_points=len(y))
