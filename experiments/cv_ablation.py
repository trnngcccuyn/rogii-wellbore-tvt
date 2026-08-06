import os,sys; 
import numpy as np, glob
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold
Xs=[];Ws=[]
for f in sorted(glob.glob('ds_*.npz')):
    d=np.load(f,allow_pickle=True); Xs.append(d['X']); Ws.append(d['well']); cols=list(d['cols'])
X=np.concatenate(Xs); W=np.concatenate(Ws); del Xs
ti=cols.index('target')
y=X[:,ti].astype(np.float64)
drop={'target'}
if len(sys.argv)>2 and sys.argv[2]=='nosurf':
    drop |= {c for c in cols if c.startswith('surf_') or c=='dp_vs_surf'}
if len(sys.argv)>2 and sys.argv[2]=='nodp':
    drop |= {c for c in cols if c.startswith('dp_') or c.startswith('tddp')}
if len(sys.argv)>2 and sys.argv[2]=='noland':
    drop |= {c for c in cols if c[:3] in ('tda','tdb','tdd','tds') and c[3:4].lstrip('-').isdigit()}
feat=[i for i,c in enumerate(cols) if c not in drop]
Xf=np.ascontiguousarray(X[:,feat]); del X
k=int(sys.argv[1])
cv=GroupKFold(5); sp=list(cv.split(Xf,y,groups=W))
tr,va=sp[k]
m=HGB(max_iter=350,learning_rate=0.06,max_leaf_nodes=63,min_samples_leaf=40,
      l2_regularization=1.0,random_state=0)
m.fit(Xf[tr],y[tr]); p=m.predict(Xf[va])
tag=sys.argv[2] if len(sys.argv)>2 else 'full'
np.savez(f'oof_{tag}_{k}.npz',p=p,y=y[va],idx=va)
print(tag,'fold',k,'rmse %.3f'%np.sqrt(np.mean((p-y[va])**2)),'base %.3f'%np.sqrt(np.mean(y[va]**2)),flush=True)
