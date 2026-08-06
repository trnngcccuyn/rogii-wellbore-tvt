import os,sys,glob; sys.path.insert(0, os.environ.get('GEOSTEER_SRC', 'src'))
import numpy as np, pandas as pd
from geosteer import SurfaceBank, geometric_exclusions, surface_transfer, dp_dip_track, beam_track, BEAMS
files=sorted(glob.glob('tr/train/*__horizontal_well.csv'))
bank=SurfaceBank(files,stride=10)
per=pd.read_csv('perwell.csv'); wids=list(per.well)
a,b=int(sys.argv[1]),int(sys.argv[2]); out='sc2.csv'
HIDE=250
for wid in wids[a:b]:
    f=f'tr/train/{wid}__horizontal_well.csv'
    h=pd.read_csv(f); tw=pd.read_csv(f.replace('horizontal_well','typewell')).sort_values('TVT')
    ti=h.TVT_input.values.astype(float); md=h.MD.values; ki=np.where(~np.isnan(ti))[0]
    if len(ki)<1100: continue
    # lateral rows only: |dTVT/dMD| small
    slope=np.abs(np.gradient(ti[ki],md[ki]))
    lat=ki[slope<0.05]
    if len(lat)<HIDE+200: continue
    hide=lat[-HIDE:]
    if hide[0]-1 not in set(ki.tolist()): continue
    h2=h.copy(); t2=ti.copy(); t2[hide]=np.nan; h2['TVT_input']=t2
    ban=geometric_exclusions(h2,bank)
    s2=surface_transfer(h2,bank,banned=ban)
    prior=(s2['surf_cons'].astype(float),np.maximum(s2['surf_spread']+s2['surf_skill'],2.0).astype(float))
    dp2,_=dp_dip_track(h2,tw,prior=prior,prior_w=1.0)
    anc=float(ti[hide[0]-1]); truth=ti[hide]
    gr=h.GR.astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw.GR)))
    ei2=np.where(np.isnan(t2))[0]
    bm=beam_track(gr.to_numpy()[ei2],tw.TVT.to_numpy(float),tw.GR.fillna(tw.GR.mean()).to_numpy(float),anc,10,20.,144.,2)
    pos={v:i for i,v in enumerate(ei2)}; sel=[pos[i] for i in hide]
    r=dict(well=wid,
        r_base=float(np.sqrt(np.mean((anc-truth)**2))),
        r_surf=float(np.sqrt(np.mean((s2['surf_cons'][hide]-truth)**2))),
        r_dp  =float(np.sqrt(np.mean((dp2[hide]-truth)**2))),
        r_beam=float(np.sqrt(np.mean((bm[sel]-truth)**2))))
    pd.DataFrame([r]).to_csv(out,mode='a',header=not os.path.exists(out),index=False)
print('done',a,b)
