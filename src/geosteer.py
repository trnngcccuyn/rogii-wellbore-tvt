"""
Merged feature builder: structural-surface transfer + GR trackers + GR misfit
landscape, all fed to a supervised meta-model that predicts
    target = TVT_eval - last_known_tvt.

Design notes
------------
* The physical identity  TVT = -Z + ANCC(x,y) + c_well  (exact to 0.05 ft on all
  485 train wells checked) means the target is the structural surface. Every
  surface feature below is a different estimator of that one quantity.
* The oracle ladder says a perfect per-well CONSTANT offset only reaches ~9.1 ft
  pooled. Level features alone cannot go below that, so the GR trackers (which
  supply SHAPE) carry the rest.
* GR misfit is strongly multi-modal. Rather than committing to an argmin we hand
  the meta-model the whole misfit landscape (offset scans) and let it decide.
"""
import os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d

FEATS_VERSION = "v2.3-2026-08-05-shadowfix"

FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


# ---------------------------------------------------------------- surface bank
class SurfaceBank:
    """Per-ROW structural surfaces of every train well, all 6 formations.

    Per-well medians (the original FormationPlaneKNN) destroy the signal: the
    surface dips ~150 ft along a single lateral.
    """
    def __init__(self, train_files, stride=10):
        self.tree = {}; self.F = {}; self.wid = []
        cx = []; cy = []
        for f in train_files:
            try:
                d = pd.read_csv(f, usecols=["X", "Y"] + FORMATIONS)
            except Exception:
                continue
            d = d.dropna()
            if len(d) < 50:
                continue
            d = d.iloc[::stride]
            w = os.path.basename(f)[:8]
            self.tree[w] = cKDTree(d[["X", "Y"]].to_numpy())
            self.F[w] = d[FORMATIONS].to_numpy(np.float32)
            self.wid.append(w); cx.append(d.X.median()); cy.append(d.Y.median())
        self.cw = np.array(self.wid); self.ctree = cKDTree(np.c_[cx, cy])

    def candidates(self, x, y, n=24):
        _, i = self.ctree.query([[np.median(x), np.median(y)]],
                                k=min(n, len(self.cw)))
        return list(self.cw[np.atleast_1d(i)[0]])

    def query(self, w, x, y):
        d, i = self.tree[w].query(np.c_[x, y], workers=-1)
        return self.F[w][i], d


def geometric_exclusions(h, bank, min_dist=50.0, n=10):
    """Drop bank wells that are the SAME borehole (id-independent guard)."""
    ev = np.isnan(h.TVT_input.values)
    x, y = h.X.values[ev], h.Y.values[ev]
    out = []
    for w in bank.candidates(x, y, n=n):
        if np.median(bank.query(w, x, y)[1]) < min_dist:
            out.append(w)
    return out


def surface_transfer(h, bank, banned=(), anchor=500, valid=400,
                     maxdist=2500., h_d=700., nb=24, top=6):
    """Neighbour-well transfer of the structural surface, anchored on the known lateral.

    For neighbour j:  TVT_j(i) = -Z(i) + ANCC_j(nearest point of j) + o_j.
    o_j is fitted on a slice of the known lateral and ABSORBS both the datum
    difference and the (near-constant) cross-dip between two parallel laterals.
    Neighbours are weighted by how well they predict a HELD-OUT slice of the
    known zone, so the weights are honest rather than in-sample.

    NOTE: only ONE formation is used. All six are constant offsets of each other
    within a well (checked: cross-formation std 1e-13), and the per-neighbour
    anchor o_j absorbs that constant, so the six transfers collapse to the same
    array (measured spread 7.7e-5 ft). The useful spread is ACROSS NEIGHBOURS.
    """
    z = h.Z.values; x = h.X.values; y = h.Y.values; ti = h.TVT_input.values
    ev = np.isnan(ti); ki = np.where(~ev)[0]; n = len(h)
    last = ti[ki[-1]] if len(ki) else np.nanmedian(ti)
    out = {"surf_cons": np.full(n, last, np.float32),
           "surf_spread": np.full(n, 25., np.float32),
           "surf_best": np.full(n, last, np.float32),
           "surf_skill": np.full(n, 25., np.float32),
           "surf_skill_best": np.full(n, 25., np.float32),
           "surf_nbdist": np.full(n, 9999., np.float32),
           "surf_nb_n": np.full(n, 0., np.float32)}
    if len(ki) < anchor + valid + 50:
        return out
    fit = ki[-(anchor + valid):-valid]; val = ki[-valid:]
    s = ti + z
    P = []; W = []; SC = []; DD = []
    for w in bank.candidates(x[ev], y[ev], n=nb):
        if w in banned:
            continue
        F, d = bank.query(w, x, y)
        md_ = float(np.median(d[ev]))
        if md_ > maxdist:
            continue
        f = F[:, 0].astype(float)
        o = np.median(s[fit] - f[fit])
        err = float(np.sqrt(np.mean(((-z[val] + f[val] + o) - ti[val]) ** 2)))
        o2 = np.median(s[val] - f[val])
        P.append((-z + f + o2).astype(np.float32)); SC.append(err); DD.append(md_)
        W.append(np.exp(-md_ / h_d) / (err ** 2 + 4.0))
    if not P:
        return out
    P = np.array(P); W = np.array(W); SC = np.array(SC); DD = np.array(DD)
    k = np.argsort(-W)[:top]
    P, W, SC, DD = P[k], W[k], SC[k], DD[k]; Wn = W / W.sum()
    mu = (P * Wn[:, None]).sum(0)
    out["surf_cons"] = mu.astype(np.float32)
    out["surf_spread"] = np.sqrt((Wn[:, None] * (P - mu) ** 2).sum(0)).astype(np.float32)
    bi = int(np.argmin(SC))
    out["surf_best"] = P[bi].astype(np.float32)
    out["surf_skill"] = np.full(n, float(np.average(SC, weights=Wn)), np.float32)
    out["surf_skill_best"] = np.full(n, float(SC[bi]), np.float32)
    out["surf_nbdist"] = np.full(n, float(np.average(DD, weights=Wn)), np.float32)
    out["surf_nb_n"] = np.full(n, float(len(P)), np.float32)
    return out


# ---------------------------------------------------------------- GR trackers
def _rnorm(v, lo=10, hi=90):
    v = np.asarray(v, float); m = np.isfinite(v)
    if m.sum() < 5:
        return np.zeros_like(v)
    a, b = np.percentile(v[m], [lo, hi])
    return (v - np.median(v[m])) / max(b - a, 1e-6)


def _prep_gr(h, tw, tw_smooth=0.5):
    T = tw.TVT.values.astype(float); G = tw.GR.values.astype(float)
    o = np.argsort(T); T, G = T[o], G[o]
    stp = max(np.median(np.diff(T)), 1e-3)
    if tw_smooth > 0:
        G = gaussian_filter1d(G, tw_smooth / stp)
    Gn = _rnorm(G); g = _rnorm(h.GR.values.astype(float))
    ti = h.TVT_input.values.astype(float); kn = ~np.isnan(ti)
    ge = np.interp(ti, T, Gn, left=np.nan, right=np.nan)
    m = np.isfinite(g) & np.isfinite(ge) & kn
    if m.sum() >= 30:
        A = np.polyfit(ge[m], g[m], 1)
        if not np.isfinite(A).all() or abs(A[0]) < 1e-3:
            A = np.array([1., 0.])
        sig = max(np.std(g[m] - np.polyval(A, ge[m])), 0.03)
    else:
        A = np.array([1., 0.]); sig = 1.0
    return T, Gn, g, A, sig


def dp_dip_track(h, tw, block_ft=300., ds=0.25, range_ft=45., eff_n=4.,
                 dip_max=0.045, dip_jump=0.006, dip_change_sd=0.0015,
                 huber=2.5, oob=6.0, prior=None, prior_w=1.0):
    """Viterbi over (TVT, stratigraphic dip).

    Decorrelated from beam search: dip is an EXPLICIT state, so the path is
    piecewise-linear in stratigraphic dip (few segments per lateral, which is
    how a human interpreter actually works) instead of a free walk.
    """
    md = h.MD.values.astype(float); ti = h.TVT_input.values.astype(float)
    ev = np.isnan(ti); ki = np.where(~ev)[0]; ei = np.where(ev)[0]
    out = ti.copy(); n = len(h)
    if len(ei) == 0 or len(ki) < 20:
        out[ev] = np.nanmedian(ti) if len(ki) == 0 else ti[ki[-1]]
        return out, 0.0
    last = ti[ki[-1]]
    T, Gn, g, A, sig = _prep_gr(h, tw)
    S = last + np.arange(-range_ft, range_ft + ds / 2, ds); nS = len(S)
    kmax = int(round(dip_max * block_ft / ds)); dips = np.arange(-kmax, kmax + 1); nD = len(dips)
    b = ((md[ei] - md[ei][0]) // block_ft).astype(int); nB = b.max() + 1
    cb = np.bincount(b, minlength=nB).astype(float)
    mdc = np.bincount(b, weights=md[ei], minlength=nB) / cb
    gr = g[ei]; ok = np.isfinite(gr)
    E = np.zeros((nB, nS))
    for j, sv in enumerate(S):
        p = A[0] * np.interp(sv, T, Gn, left=np.nan, right=np.nan) + A[1]
        r = np.where(ok & np.isfinite(p), gr - p, np.nan)
        c = np.minimum((r / sig) ** 2, huber ** 2)
        c = np.where(np.isfinite(c), c, oob)
        E[:, j] = np.bincount(b, weights=c, minlength=nB) / cb * eff_n
    if prior is not None and prior_w > 0:
        mu, sdv = prior
        mub = np.bincount(b, weights=mu[ei], minlength=nB) / cb
        sdb = np.bincount(b, weights=sdv[ei], minlength=nB) / cb
        E += prior_w * ((S[None, :] - mub[:, None]) / np.maximum(sdb[:, None], 0.5)) ** 2
    tail = ki[-600:]
    d0 = np.polyfit(md[tail], ti[tail], 1)[0] if len(tail) >= 20 else 0.
    d0i = int(np.clip(round(d0 * block_ft / ds), -kmax, kmax))
    INF = 1e12
    D = np.full((nD, nS), INF); ptr = np.zeros((nB, nD, nS), np.int8)
    for di in range(nD):
        D[di] = E[0] + ((S - last) / 1.5) ** 2 + (((dips[di] - d0i) * ds / block_ft) / 0.01) ** 2
    J = max(1, int(round(dip_jump * block_ft / ds))); dch = np.arange(-J, J + 1)
    dcost = ((dch * ds / block_ft) / dip_change_sd) ** 2
    for t in range(1, nB):
        new = np.full((nD, nS), INF); narg = np.zeros((nD, nS), np.int8)
        for di in range(nD):
            k = dips[di]; best = np.full(nS, INF); barg = np.zeros(nS, np.int8)
            for ci, cc in enumerate(dch):
                s_ = di - cc
                if s_ < 0 or s_ >= nD:
                    continue
                src = np.roll(D[s_], k) + dcost[ci]
                if k > 0: src[:k] = INF
                elif k < 0: src[k:] = INF
                u = src < best; best[u] = src[u]; barg[u] = cc
            new[di] = best + E[t]; narg[di] = barg
        D = new; ptr[t] = narg
    di, si = np.unravel_index(np.argmin(D), D.shape)
    path = np.zeros(nB, int); dpath = np.zeros(nB, int); path[-1] = si; dpath[-1] = di
    for t in range(nB - 1, 0, -1):
        c = int(ptr[t, dpath[t], path[t]])
        path[t - 1] = path[t] - dips[dpath[t]]; dpath[t - 1] = dpath[t] - c
    tb = S[np.clip(path, 0, nS - 1)]
    P = np.exp(-(D - D.min()) / 2.).sum(0); P /= P.sum()
    mn = (P * S).sum(); spread = float(np.sqrt((P * (S - mn) ** 2).sum()))
    out[ei] = np.interp(md[ei], mdc, tb)
    return out, spread


def beam_track(gr_h, tw_tvt, tw_gr, start_tvt, bs=10, mc=20., es=144., r=2):
    """Numpy beam search over the typewell index (no numba needed)."""
    n = len(gr_h); nt = len(tw_tvt)
    if n == 0:
        return np.array([start_tvt])
    s = pd.Series(gr_h).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    sgr = (s.rolling(2 * r + 1, center=True, min_periods=1).mean() if r > 0 else s).to_numpy()
    si = int(np.argmin(np.abs(tw_tvt - start_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2]); MC = mc * np.array([2., 1., 0., 1., 2.])
    bidx = np.full(bs, si, np.int64); bcost = np.full(bs, np.inf); bcost[0] = 0.; bn = 1
    res = np.zeros(n)
    for step in range(n):
        ni = bidx[:bn, None] + MOVES[None, :]
        valid = (ni >= 0) & (ni < nt); ci = np.clip(ni, 0, nt - 1)
        tot = bcost[:bn, None] + (sgr[step] - tw_gr[ci]) ** 2 / es + MC[None, :]
        tot = np.where(valid, tot, np.inf)
        nf = ni.ravel()[valid.ravel()]; tf = tot.ravel()[valid.ravel()]
        if len(nf) == 0:
            res[step:] = tw_tvt[bidx[0]]; break
        order = np.argsort(tf); ns_, ts_ = nf[order], tf[order]
        _, first = np.unique(ns_, return_index=True)
        nu, tu = ns_[first], ts_[first]
        kept = min(bs, len(nu)); top = np.argsort(tu)[:kept]
        bidx[:kept] = nu[top]; bcost[:kept] = tu[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]; bcost[kept:] = np.inf
        bn = kept; res[step] = tw_tvt[bidx[0]]
    return res


def multi_scale_ncc(kgr, ktvt, hgr, hws=(8, 15, 25), stride=3):
    """Match a window of eval GR against the KNOWN-zone GR (whose TVT is known).

    Uses the well's own log as the reference instead of the vertical typewell:
    same tool, same hole angle, same mud, so the calibration problem disappears.
    """
    out = []
    for hwin in hws:
        win = 2 * hwin + 1; nk = len(kgr); nh = len(hgr)
        if nk < win + 1 or nh == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        kg = pd.Series(kgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
        hg = pd.Series(hgr).rolling(5, center=True, min_periods=1).mean().to_numpy(np.float32)
        sts = np.arange(0, nk - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            out.append((np.full(nh, ktvt[-1], np.float32), np.zeros(nh, np.float32))); continue
        C = kg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        hp = np.pad(hg, hwin, mode="edge")
        H = hp[np.arange(nh)[:, None] + np.arange(win)[None, :]]
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = Hn @ Cn.T / win
        out.append((ktvt[np.clip(sts[ncc.argmax(1)] + hwin, 0, nk - 1)].astype(np.float32),
                    ncc.max(1).astype(np.float32)))
    tv = np.stack([o[0] for o in out], 1); sc = np.stack([o[1] for o in out], 1)
    sw = np.exp(3. * sc); sw /= sw.sum(1, keepdims=True) + 1e-9
    return out, (tv * sw).sum(1).astype(np.float32)


BEAMS = [(10, 20., 144., 2, "cons"), (10, 8., 64., 2, "loose"),
         (8, 35., 220., 1, "vcons"), (10, 14., 90., 5, "sm5"),
         (20, 4., 36., 3, "vloose")]

PF_PARTICLES = 300
PF_SEEDS = 16      # 32 seeds measured no better and cost 2x

ANCH_OFFS = np.array([-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80], np.float32)
TRK_OFFS  = np.array([-30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30], np.float32)


def build_well(hw_path, tw_path, is_train, bank):
    """One well -> a DataFrame of eval-row features (+ target when training)."""
    wid = os.path.basename(hw_path)[:8]
    try:
        h = pd.read_csv(hw_path); tw = pd.read_csv(tw_path).sort_values("TVT")
    except Exception:
        return None
    ti = h.TVT_input.values.astype(float)
    ev = np.isnan(ti); ki = np.where(~ev)[0]; ei = np.where(ev)[0]
    if len(ei) == 0 or len(ki) < 50 or len(tw) < 3:
        return None
    if is_train and ("TVT" not in h.columns or h.TVT.isna().all()):
        return None
    last = float(ti[ki[-1]]); z = h.Z.values.astype(float); md = h.MD.values.astype(float)
    tw_tvt = tw.TVT.to_numpy(float); tw_gr = tw.GR.fillna(tw.GR.mean()).to_numpy(float)

    banned = geometric_exclusions(h, bank)          # leak guard, id-independent
    surf = surface_transfer(h, bank, banned=banned)

    prior = (surf["surf_cons"].astype(float),
             np.maximum(surf["surf_spread"].astype(float) + surf["surf_skill"].astype(float), 2.0))
    dp_a, dp_sp = dp_dip_track(h, tw, prior=prior, prior_w=1.0)
    dp_b, _     = dp_dip_track(h, tw, block_ft=200., eff_n=8., prior=prior, prior_w=3.0)
    pf = lik_pf(h, tw, n_particles=PF_PARTICLES, n_seeds=PF_SEEDS)

    gr_full = h.GR.astype(float).interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.to_numpy(np.float32)[ei]
    # BUG FIX: restrict the NCC reference and the slope fit to the LATERAL part of
    # the known zone. Including the build section (where |dTVT/dMD| is huge) made
    # sc_ens_d land ~92 ft off with 26% of rows beyond +-200 ft, and made the
    # linear-extrapolation feature median +945 ft. Same root cause in both.
    kslope = np.abs(np.gradient(ti[ki], md[ki]))
    klat = ki[kslope < 0.05]
    if len(klat) < 60:
        klat = ki[-min(len(ki), 400):]
    kgr = gr_full.to_numpy(np.float32)[klat]
    ktvt = ti[klat].astype(np.float32)

    beams = {t: beam_track(hgr, tw_tvt, tw_gr, last, bs, mc, es, r)
             for (bs, mc, es, r, t) in BEAMS}
    beam_ref = (beams["cons"] + beams["sm5"]) / 2.
    _, sc_ens = multi_scale_ncc(kgr, ktvt, hgr)

    ne = len(ei)
    def col(v): return np.full(ne, np.float32(v), np.float32)
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    A = np.polyfit(tw_at_k, kgr, 1) if np.std(tw_at_k) > 1e-6 else np.array([1., 0.])
    lkX, lkY, lkZ, lkMD = h.X.values[ki[-1]], h.Y.values[ki[-1]], z[ki[-1]], md[ki[-1]]
    md_since = (md[ei] - lkMD).astype(np.float32)
    slp_all = np.polyfit(md[klat], ti[klat], 1)[0]
    slp_50  = np.polyfit(md[ki[-50:]], ti[ki[-50:]], 1)[0] if len(ki) >= 50 else slp_all
    slp_all = float(np.clip(slp_all, -0.05, 0.05))
    slp_50  = float(np.clip(slp_50,  -0.05, 0.05))
    slp_b   = (last + slp_all * md_since).astype(np.float32)

    F = {"well": wid, "id": [f"{wid}_{i}" for i in ei], "last_known_tvt": col(last)}
    # --- structural surface family (level) ---
    LEVEL = ("surf_cons", "surf_best")
    for k, v in surf.items():
        vv = (v[ei] - last) if k in LEVEL else v[ei]
        F[k] = (np.clip(vv, -150., 150.) if k in LEVEL else vv).astype(np.float32)
    # --- tracker family (shape) ---
    for k, v in pf.items():
        F[k + "_d"] = (v[ei] - last).astype(np.float32)
    pfk = [k for k in pf if k.startswith("pf_scale_")]
    PFm = np.stack([pf[k][ei] for k in pfk], 0)
    F["pf_scale_std"] = PFm.std(0).astype(np.float32)
    F["pf_mix"] = ((0.5 * (dp_a[ei] + dp_b[ei]) * 0.5 + 0.5 * pf["pf_scale_8"][ei]) - last).astype(np.float32)
    F["pf_vs_dp"] = (pf["pf_scale_8"][ei] - 0.5 * (dp_a[ei] + dp_b[ei])).astype(np.float32)
    F["dp_a_d"] = (dp_a[ei] - last).astype(np.float32)
    F["dp_b_d"] = (dp_b[ei] - last).astype(np.float32)
    F["dp_spread"] = col(dp_sp)
    F["dp_vs_surf"] = (dp_a[ei] - surf["surf_cons"][ei]).astype(np.float32)
    for t, p in beams.items():
        F[f"beam_{t}_d"] = (p - last).astype(np.float32)
    B = np.stack([p - last for p in beams.values()], 1)
    F["beam_mean_d"] = B.mean(1).astype(np.float32)
    F["beam_std_d"]  = B.std(1).astype(np.float32)
    F["sc_ens_d"] = (sc_ens - last).astype(np.float32)
    F["sc_vs_beam"] = (sc_ens - beams["cons"]).astype(np.float32)
    # TVT - last_known has p99 ~56 ft and absmax ~84 across train, so a linear
    # extrapolation reaching +-460 ft is noise, not signal. Clip to the physical range.
    F["slp_b_d"] = np.clip(slp_b - last, -80., 80.).astype(np.float32)
    # --- GR misfit LANDSCAPE (not an argmin): let the model pick the mode ---
    for o in ANCH_OFFS:
        F[f"tda{int(o)}"] = (hgr - np.polyval(A, np.interp(last + o, tw_tvt, tw_gr))).astype(np.float32)
    for o in TRK_OFFS:
        F[f"tdbc{int(o)}"] = (hgr - np.polyval(A, np.interp(beam_ref + o, tw_tvt, tw_gr))).astype(np.float32)
        F[f"tddp{int(o)}"] = (hgr - np.polyval(A, np.interp(dp_a[ei] + o, tw_tvt, tw_gr))).astype(np.float32)
        F[f"tdsf{int(o)}"] = (hgr - np.polyval(A, np.interp(surf["surf_cons"][ei] + o, tw_tvt, tw_gr))).astype(np.float32)
        F[f"tdpf{int(o)}"] = (hgr - np.polyval(A, np.interp(pf["pf_scale_8"][ei] + o, tw_tvt, tw_gr))).astype(np.float32)
    # --- GR shape + geometry ---
    gs = pd.Series(gr_full.values)
    for w in (5, 21, 51, 101):
        r = gs.rolling(w, center=True, min_periods=1)
        F[f"grm{w}"] = r.mean().to_numpy(np.float32)[ei]
        F[f"grs{w}"] = r.std().fillna(0).to_numpy(np.float32)[ei]
    F["gr"] = hgr
    F["gr_d1"] = gs.diff().fillna(0.).to_numpy(np.float32)[ei]
    F["gr_d2"] = gs.diff().diff().fillna(0.).to_numpy(np.float32)[ei]
    mdd = h.MD.diff().replace(0, np.nan)
    F["dzdmd"] = (h.Z.diff() / mdd).to_numpy(np.float32)[ei]
    F["dxdmd"] = (h.X.diff() / mdd).to_numpy(np.float32)[ei]
    F["dydmd"] = (h.Y.diff() / mdd).to_numpy(np.float32)[ei]
    frac = (np.arange(ne) / max(ne - 1, 1)).astype(np.float32)
    F.update(md_since=md_since, frac=frac, frac2=frac ** 2, sqrt_frac=np.sqrt(frac),
             z=z[ei].astype(np.float32),
             dz=(z[ei] - lkZ).astype(np.float32),
             dxy=np.hypot(h.X.values[ei] - lkX, h.Y.values[ei] - lkY).astype(np.float32),
             known_len=col(len(ki)), eval_len=col(ne),
             slp_all=col(slp_all), slp_50=col(slp_50),
             cal_a=col(A[0]), cal_b=col(A[1]),
             ktvt_std=col(float(ktvt.std())), ktvt_range=col(float(np.ptp(ktvt))),
             tw_range=col(float(np.ptp(tw_tvt))))
    out = pd.DataFrame(F)
    if is_train:
        out["target"] = (h.TVT.to_numpy(np.float32)[ei] - np.float32(last))
    return out


# ------------------------------------------------- likelihood-weighted PF bank
def lik_pf(h, tw, n_particles=300, n_seeds=32, scales=(3., 5., 8., 12.),
           init_spread=4.5, mom=0.998, vel_noise=0.002, pos_noise=0.005,
           rough_p=0.1, rough_v=0.001, resamp=0.5, seed_base=0):
    """128-seed style likelihood-weighted particle filter, vectorised over
    (seed x particle) so it runs without numba.

    State is the stratigraphic level U = TVT + Z with a random-walk rate. Each
    seed is scored by its total log-likelihood, and seeds are combined with a
    softmax at several temperatures -- a low temperature trusts the single best
    run, a high one averages. Decorrelated from the DP tracker: stochastic and
    forward-only versus deterministic global optimisation.
    """
    ti = h.TVT_input.values.astype(float); z = h.Z.values.astype(float)
    md = h.MD.values.astype(float)
    ev = np.isnan(ti); ki = np.where(~ev)[0]; ei = np.where(ev)[0]
    n = len(h)
    if len(ei) == 0 or len(ki) < 20:
        base = ti[ki[-1]] if len(ki) else np.nanmedian(ti)
        return {f"pf_scale_{s:g}": np.full(n, base) for s in scales}
    last = ti[ki[-1]]
    T = tw.TVT.values.astype(float); G = tw.GR.fillna(tw.GR.mean()).values.astype(float)
    o = np.argsort(T); T, G = T[o], G[o]
    gk = h.GR.values.astype(float)[ki]
    twk = np.interp(ti[ki], T, G)
    m = np.isfinite(gk) & np.isfinite(twk)
    gs = float(np.clip(np.std(gk[m] - twk[m]) if m.sum() > 20 else 30., 10., 60.))
    # initial rate from the lateral tail only
    ks = np.abs(np.gradient(ti[ki], md[ki])); klat = ki[ks < 0.05]
    tail = klat[-30:] if len(klat) >= 30 else ki[-30:]
    dt = np.diff(ti[tail] + z[tail]); dm = np.diff(md[tail]); mm = dm > 0
    ir = float(np.median(dt[mm] / dm[mm])) if mm.sum() >= 3 else 0.

    gr = pd.Series(h.GR.values).interpolate(limit_direction="both").fillna(G.mean()).to_numpy()[ei]
    mdv = md[ei]; zv = z[ei]
    S, N = n_seeds, n_particles
    rng = np.random.default_rng(seed_base)
    pos = last + z[ki[-1]] + init_spread * rng.standard_normal((S, N))
    rate = ir + 0.01 * rng.standard_normal((S, N))
    w = np.full((S, N), 1. / N)
    ll = np.zeros(S)
    est = np.empty((S, len(ei)))
    lo, hi = T[0] - 100., T[-1] + 100.
    prev = md[ki[-1]]
    for i in range(len(ei)):
        dm_ = max(mdv[i] - prev, 1.0); prev = mdv[i]
        rate = mom * rate + vel_noise * rng.standard_normal((S, N))
        pos = pos + rate * dm_ + pos_noise * rng.standard_normal((S, N))
        tvt = np.clip(pos - zv[i], lo, hi); pos = tvt + zv[i]
        d = (gr[i] - np.interp(tvt, T, G)) / gs
        lk = np.exp(-0.5 * np.minimum(d * d, 600.))
        avg = np.maximum((w * lk).sum(1), 1e-300)
        ll += np.log(avg)
        w = w * lk
        ws = w.sum(1, keepdims=True)
        w = np.where(ws > 0, w / np.maximum(ws, 1e-300), 1. / N)
        neff = 1. / np.maximum((w ** 2).sum(1), 1e-300)
        need = np.where(neff < resamp * N)[0]
        if len(need):
            cum = np.cumsum(w[need], axis=1)
            u = (rng.uniform(0, 1. / N, (len(need), 1)) + np.arange(N) / N)
            idx = np.clip(np.array([np.searchsorted(cum[j], u[j]) for j in range(len(need))]), 0, N - 1)
            pos[need] = np.take_along_axis(pos[need], idx, 1) + rough_p * rng.standard_normal((len(need), N))
            rate[need] = np.take_along_axis(rate[need], idx, 1) + rough_v * rng.standard_normal((len(need), N))
            w[need] = 1. / N
        est[:, i] = (w * (pos - zv[i])).sum(1)
    lln = ll - ll.max()
    out = {}
    for sc in scales:
        ww = np.exp(lln / float(sc)); ww /= ww.sum()
        a = np.full(n, last); a[ei] = (ww[:, None] * est).sum(0)
        out[f"pf_scale_{sc:g}"] = a
    a = np.full(n, last); a[ei] = est.mean(0); out["pf_mean"] = a
    return out

print("feature module loaded:", FEATS_VERSION)
