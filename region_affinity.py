"""Learned linking of visible pieces of the same occluded object."""
import numpy as np
from scipy.spatial import cKDTree


def region_pairs(r, grid):
    """Symmetric pair features from observed cells only; labels never enter features."""
    ids=np.unique(grid[(grid>1)&r['valid']]);regions=[]
    for label in ids:
        yy,xx=np.where((grid==label)&r['valid'])
        # Cap deterministic plane-fit samples without changing scene scale.
        pick=np.linspace(0,len(xx)-1,min(len(xx),1500),dtype=int)
        yy=yy[pick];xx=xx[pick]
        means=r['means'][yy,xx]
        xyz=np.column_stack((r['lo'][0]+xx*r['cell'],r['lo'][1]+yy*r['cell'],means[:,0]))
        center=xyz.mean(0);delta=xyz-center
        eig,vec=np.linalg.eigh(delta.T@delta/max(len(delta),1))
        normal=vec[:,0];direction=vec[:,-1]
        regions.append(dict(center=center,normal=normal,direction=direction,rms=np.sqrt(max(eig[0],0)),
                            color=means[:,1:4].mean(0),color_std=means[:,1:4].std(0),
                            size=int(np.sum((grid==label)&r['valid'])),extent=np.sqrt(np.maximum(eig,0)),
                            lo=xyz.min(0),hi=xyz.max(0),sample=xyz[::max(1,len(xyz)//150)]))
    pairs=[];features=[]
    for i,a in enumerate(regions):
        for j in range(i+1,len(regions)):
            b=regions[j];d=b['center']-a['center']
            plane=sorted((abs(d@a['normal']),abs(d@b['normal'])))
            gap=np.maximum(np.maximum(a['lo']-b['hi'],b['lo']-a['hi']),0)
            near=float(cKDTree(a['sample']).query(b['sample'])[0].min())
            f=[*np.abs(d),np.linalg.norm(d),*plane,abs(a['normal']@b['normal']),
               abs(a['direction']@b['direction']),*gap,near,
               min(a['rms'],b['rms']),max(a['rms'],b['rms']),
               *np.abs(a['color']-b['color']),*(a['color_std']+b['color_std']),
               *np.minimum(a['extent'],b['extent']),*np.maximum(a['extent'],b['extent']),
               np.log1p(min(a['size'],b['size'])),np.log1p(max(a['size'],b['size'])),
               min(a['size'],b['size'])/max(a['size'],b['size'])]
            pairs.append((i,j));features.append(f)
    return ids,np.asarray(pairs,dtype=int).reshape(-1,2),np.asarray(features,dtype=np.float32)


def merge_regions(grid, ids, pairs, scores, threshold):
    """Complete-link clustering: every pair across a merge must be confident."""
    n=len(ids);matrix=np.eye(n);groups=[{i} for i in range(n)]
    for (i,j),s in zip(pairs,scores):matrix[i,j]=matrix[j,i]=s
    while True:
        best=None;value=threshold
        for i,a in enumerate(groups):
            for j in range(i+1,len(groups)):
                confidence=float(matrix[np.ix_(sorted(a),sorted(groups[j]))].min())
                if confidence>value:best=(i,j);value=confidence
        if best is None:break
        i,j=best;groups[i]|=groups[j];groups.pop(j)
    out=np.ones_like(grid)
    for label,group in enumerate(groups,2):out[np.isin(grid,ids[sorted(group)])]=label
    return out


def link_prediction(result, classifier, threshold):
    labels,review,r,grid,info=result
    ids,pairs,x=region_pairs(r,grid)
    if not len(pairs):return result
    scores=classifier.predict_proba(x)[:,list(classifier.classes_).index(1)]
    linked=merge_regions(grid,ids,pairs,scores,threshold)
    changed=linked.ravel()[r['flat']]
    # Flag every linked region for review, including the piece whose ID survived.
    group_count={lab:len(np.unique(grid[linked==lab])) for lab in np.unique(linked[linked>1])}
    linked_mask=np.isin(linked,[lab for lab,n in group_count.items() if n>1])
    review=np.maximum(review,linked_mask.ravel()[r['flat']].astype(np.float32))
    info=dict(info,objects=len(np.unique(linked[linked>1])),merged_pieces=int(len(ids)-len(np.unique(linked[linked>1]))),
              review_fraction=float(review.mean()),region_link_threshold=threshold)
    return changed,review,r,linked,info
