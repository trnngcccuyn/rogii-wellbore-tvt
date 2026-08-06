import os,sys,time; 
import numpy as np, glob
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.model_selection import GroupKFold
Xs=[];Ws=[]
for f in sorted(glob.glob('ds_*.npz')):
    d=np.load(f,allow_pickle=True); Xs.append(d['X']); Ws.append(d['well']); cols=list(d['cols'])
X=np.concatenate(Xs); W=np.concatenate(Ws); del Xs
y=X[:,cols.index('target')].astype(float)
feat=[i for i,c in enumerate(cols) if c!='target']
Xf=np.ascontiguousarray(X[:,feat]); del X
step=int(sys.argv[1]); k=int(sys.argv[2])
sp=list(GroupKFold(5).split(Xf,y,groups=W)); tr,va=sp[k]
tr=tr[::step]
t0=time.time()
m=HGB(max_iter=350,learning_rate=.06,max_leaf_nodes=63,min_samples_leaf=40,
      l2_regularization=1.,random_state=0)
m.fit(Xf[tr],y[tr]); p=m.predict(Xf[va])
print('thin=%d fold%d train_rows=%d rmse %.3f  fit %.0fs'%(step,k,len(tr),np.sqrt(np.mean((p-y[va])**2)),time.time()-t0),flush=True)
