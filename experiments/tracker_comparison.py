import os,sys,time; sys.path.insert(0, os.environ.get('GEOSTEER_SRC', 'src'))
import numpy as np, pandas as pd, glob
from geosteer import SurfaceBank, lik_pf, dp_dip_track, geometric_exclusions, surface_transfer
files=sorted(glob.glob('tr/train/*__horizontal_well.csv'))
bank=SurfaceBank(files,stride=10)
rng=np.random.default_rng(11); sel=list(rng.choice(len(files),160,replace=False))
a,b=int(sys.argv[1]),int(sys.argv[2]); out='cmp3.csv'
for i in sel[a:b]:
    f=files[i]; h=pd.read_csv(f); tw=pd.read_csv(f.replace('horizontal_well','typewell')).sort_values('TVT')
    if 'TVT' not in h or h.TVT.isna().all(): continue
    ev=h.TVT_input.isna().values
    if ev.sum()<100: continue
    ban=geometric_exclusions(h,bank); s=surface_transfer(h,bank,banned=ban)
    pr=(s['surf_cons'].astype(float),np.maximum(s['surf_spread']+s['surf_skill'],2.0).astype(float))
    dpa,_=dp_dip_track(h,tw,prior=pr,prior_w=1.0)
    dpb,_=dp_dip_track(h,tw,block_ft=200.,eff_n=8.,prior=pr,prior_w=3.0)
    pf=lik_pf(h,tw)['pf_scale_8']
    t=h.TVT.values[ev]; bs=h.TVT_input.values[~ev][-1]
    S=s['surf_cons'][ev].astype(float); D=(0.5*(dpa+dpb))[ev]; P=pf[ev]
    e=lambda v: float(np.sqrt(np.mean((v-t)**2)))
    M=np.stack([S,D,P])
    r=dict(well=os.path.basename(f)[:8],n=int(ev.sum()),base=e(np.full(len(t),bs)),
      surf=e(S),dp=e(D),pf=e(P),
      mix2=e(.5*D+.5*P), mean3=e(M.mean(0)), med3=e(np.median(M,0)),
      w442=e(.4*S+.4*D+.2*P), w433=e(.4*S+.3*D+.3*P), w334=e(.3*S+.3*D+.4*P),
      trim=e(0.5*np.median(M,0)+0.5*M.mean(0)))
    pd.DataFrame([r]).to_csv(out,mode='a',header=not os.path.exists(out),index=False)
print('done',a,b)
