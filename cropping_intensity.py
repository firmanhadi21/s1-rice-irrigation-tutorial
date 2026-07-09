#!/usr/bin/env python3
"""
Cycle-based paddy mask + cropping intensity (CI) for the MT 2024/25 agricultural
year, with a Model-B rice candidate for specificity.

Logic (see PADDY_RG_SG_MULTIYEAR_STATUS.md):
- The agricultural year (Musim Tanam) runs ~Oct->Sep; MT 2024/25 is the one
  COMPLETE ag-year in our data = continuous combined-stack bands 25..56.
- CI = number of COMPLETE rice crop cycles (flood-trough -> canopy-peak swings
  with amplitude >= MIN_RISE) in that window. This is data-driven (no fixed
  calendar cutoff, no absolute-dB threshold) and rejects the transient false
  positives that inflate the period-count consensus.
- paddy_mask = (n_cycles >= 1)  AND  rice candidate (Model B flagged the pixel
  paddy in >= 1 MT-window period). The candidate adds rice-specificity (a
  non-rice crop can also show a canopy rise; only the model knows it's rice).
- cropping_intensity = n_cycles (reported 1 / 2 / 3+).

Vectorized cycle counter: per-pixel state machine over the 32 time steps, run on
all pixels at once (per-pixel find_peaks over 56M pixels would take hours).

Outputs (in --output dir):
  cropping_intensity.tif  (uint8: 0 non-paddy, 1/2/3 crops, 3 = 3+)
  paddy_mask.tif          (uint8: 1 paddy, 0 else)
  statistics.txt
"""
import argparse
import os
import math
import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from scipy.signal import savgol_filter

STACK = '/home/unika_sianturi/work/rice-growth-stage-mapping/stacks/java_vh_2024_2026_50m.tif'
MASK = '/home/unika_sianturi/work/rice-growth-stage-mapping/data/java_mask_rg_50m.tif'
CIMAP = 'data/Open-SEA-Rice-10_Java.tif'
MT_BANDS = list(range(25, 57))     # MT 2024/25 (continuous): 2024 p25 -> 2025 p24
CLIP_LO, CLIP_HI = -3500.0, -100.0
MIN_RISE = 400.0                   # dBx100 (4 dB): validated operating point
SG_WIN, SG_POLY = 5, 2


def smooth_clip(vh):
    """vh: (N,T) int/float -> clipped + SG-smoothed float32."""
    vh = np.clip(vh.astype(np.float32), CLIP_LO, CLIP_HI)
    if vh.shape[1] >= SG_WIN:
        vh = savgol_filter(vh, SG_WIN, SG_POLY, axis=1).astype(np.float32)
    return vh


def detect_anomalous_bands(stack_path, mask_path, bands, dec=10, near0=50.0, frac=0.5):
    """Find ~0 dB artefact bands within `bands` (e.g. SNAP gaps stored as ~0 dBx100).
    A band is anomalous if >`frac` of land pixels have |value| < `near0`.
    Returns indices INTO `bands` (i.e. column positions in the MT window)."""
    from rasterio.enums import Resampling
    with rasterio.open(mask_path) as m:
        Hc, Wc = m.height // dec, m.width // dec
        land = m.read(1, out_shape=(Hc, Wc), resampling=Resampling.nearest) > 0
    bad = []
    with rasterio.open(stack_path) as s:
        nod = s.nodata
        for j, b in enumerate(bands):
            a = s.read(b, out_shape=(Hc, Wc), resampling=Resampling.nearest).astype(np.float32)
            v = a[land]
            v = v[v != nod] if nod is not None else v
            if v.size and np.mean(np.abs(v) < near0) > frac:
                bad.append(j)
    return bad


def interp_bad_cols(vh, bad_cols):
    """Linearly interpolate the given column indices of vh (N,T) from their nearest
    good (non-bad) neighbours, in place. Handles consecutive runs and edges."""
    if not bad_cols:
        return vh
    T = vh.shape[1]
    badset = set(bad_cols)
    good = [c for c in range(T) if c not in badset]
    if not good:
        return vh
    for c in sorted(bad_cols):
        left = max((g for g in good if g < c), default=None)
        right = min((g for g in good if g > c), default=None)
        if left is None:
            vh[:, c] = vh[:, right]
        elif right is None:
            vh[:, c] = vh[:, left]
        else:
            w = (c - left) / (right - left)
            vh[:, c] = vh[:, left] * (1 - w) + vh[:, right] * w
    return vh


def count_cycles_vec(vh, min_rise=MIN_RISE):
    """Vectorized complete-cycle count. vh: (N,T) smoothed.
    Per-pixel state machine:
      state 0 = tracking the running MIN (flood), fire a crop when value rises
                >= min_rise above that min;
      state 1 = tracking the running MAX (canopy), re-arm (back to state 0) when
                value drops >= min_rise below that max (harvest/next flood).
    A crop requires a full flood->canopy rise; the next crop requires a
    canopy->trough drop first -> counts COMPLETE cycles only.
    Returns (N,) int16 cycle counts.
    """
    N, T = vh.shape
    cycles = np.zeros(N, dtype=np.int16)
    ref = vh[:, 0].astype(np.float32)          # running min (state0) / max (state1)
    state = np.zeros(N, dtype=np.int8)
    for t in range(1, T):
        v = vh[:, t]
        s0 = state == 0
        # state 0: track min, fire on rise
        ref = np.where(s0, np.minimum(ref, v), ref)
        fire = s0 & ((v - ref) >= min_rise)
        cycles += fire
        state = np.where(fire, np.int8(1), state)
        ref = np.where(fire, v, ref)            # begin tracking canopy max
        # state 1: track max, re-arm on drop
        s1 = state == 1
        ref = np.where(s1, np.maximum(ref, v), ref)
        rearm = s1 & ((ref - v) >= min_rise)
        state = np.where(rearm, np.int8(0), state)
        ref = np.where(rearm, v, ref)           # begin tracking next flood min
    return cycles


def validate(per_class=400, seed=42):
    """Check vectorized counter vs Open-SEA single/double/triple classes."""
    rng = np.random.default_rng(seed)
    ci = rasterio.open(CIMAP); ci_arr = ci.read(1)
    st = rasterio.open(STACK); nodata = st.nodata
    print(f"MT 2024/25 bands {MT_BANDS[0]}..{MT_BANDS[-1]} | min_rise={MIN_RISE}")
    for cls in (1, 2, 3):
        rs, cs = np.where(ci_arr == cls)
        pick = rng.choice(len(rs), size=min(per_class * 3, len(rs)), replace=False)
        series = []
        got = 0
        for k in pick:
            if got >= per_class:
                break
            x, y = xy(ci.transform, rs[k], cs[k])
            r, c = rowcol(st.transform, x, y); r, c = int(r), int(c)
            if not (0 <= r < st.height and 0 <= c < st.width):
                continue
            s = np.array([st.read(b, window=((r, r+1), (c, c+1)))[0, 0] for b in MT_BANDS], float)
            if nodata is not None and np.any(s == nodata):
                continue
            series.append(s); got += 1
        arr = np.array(series)
        n = count_cycles_vec(smooth_clip(arr))
        print(f"  Open-SEA CI={cls}: n={len(n)} mean detected={n.mean():.2f} "
              f"dist={np.bincount(np.clip(n,0,4),minlength=5).tolist()}")


def build_candidate(pred_dir, periods_by_year, conf_thr=0.5):
    """Union mask: pixel flagged paddy (==1) with confidence>=conf_thr in >=1 of
    the given MT-window prediction periods. Adds rice-specificity."""
    cand = None
    for year, periods in periods_by_year.items():
        for p in periods:
            pf = f"{pred_dir}/{year}/period_{p:02d}/paddy_predictions.tif"
            cf = f"{pred_dir}/{year}/period_{p:02d}/confidence.tif"
            if not (os.path.exists(pf) and os.path.exists(cf)):
                continue
            pa = rasterio.open(pf).read(1)
            co = rasterio.open(cf).read(1)
            hit = (pa == 1) & (co >= conf_thr)
            cand = hit if cand is None else (cand | hit)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['validate', 'full'], default='validate')
    ap.add_argument('--output', default='cropping_intensity_mt2024_25')
    ap.add_argument('--pred-dir', default='predictions_paddy_survey')
    ap.add_argument('--no-candidate', action='store_true',
                    help='skip Model-B rice candidate intersection (cycles only)')
    ap.add_argument('--candidate-mask', default=None,
                    help='use this raster (>0) as the rice mask instead of the Model-B '
                         'union, e.g. data/LBS_Jawa_50m_matched.tif (authoritative baku sawah)')
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--stack', default=None, help='override VH stack path (default: hardcoded HPC path)')
    ap.add_argument('--mask', default=None, help='override land mask path')
    ap.add_argument('--cimap', default=None, help='override reference CI map (validate mode)')
    args = ap.parse_args()

    global STACK, MASK, CIMAP
    if args.stack: STACK = args.stack
    if args.mask: MASK = args.mask
    if args.cimap: CIMAP = args.cimap

    if args.mode == 'validate':
        validate()
        return

    st = rasterio.open(STACK)
    H, W = st.height, st.width
    nodata = st.nodata
    land = rasterio.open(MASK).read(1) > 0
    print(f"land pixels: {int(land.sum()):,}")

    # Detect ~0 dB artefact bands inside the MT window and interpolate them out of
    # the VH series before cycle counting. Two consecutive ~0 dB bands otherwise
    # fake a flood->canopy swing and inflate the cropping index. For MT 2024/25
    # these are 2025 p2,p3 (combined bands 33,34 = window positions 8,9).
    bad_cols = detect_anomalous_bands(STACK, MASK, MT_BANDS)
    bad_bands = [MT_BANDS[j] for j in bad_cols]
    bad_names = [st.descriptions[b - 1] for b in bad_bands]
    print(f"anomalous (~0 dB) bands interpolated: {bad_bands} {bad_names} "
          f"(window positions {bad_cols})")

    # rice candidate: authoritative LBS mask (preferred), or Model-B union, or none
    candidate = None
    if args.candidate_mask:
        print(f"using candidate mask: {args.candidate_mask}")
        candidate = rasterio.open(args.candidate_mask).read(1) > 0
        print(f"candidate (mask & land) pixels: {int((candidate & land).sum()):,}")
    elif not args.no_candidate:
        # MT 2024/25 window covered by available predictions:
        # 2024 p25-31 and 2025 p7-30 (p1-6 not predicted; p8,p9 anomalous -> drop)
        periods_by_year = {2024: list(range(25, 32)),
                           2025: [p for p in range(7, 31) if p not in (8, 9)]}
        print("building Model-B rice candidate union...")
        candidate = build_candidate(args.pred_dir, periods_by_year)
        print(f"candidate paddy pixels: {int((candidate & land).sum()):,}")

    ci_out = np.zeros((H, W), dtype=np.uint8)

    nb = (H + args.chunk - 1) // args.chunk
    for i in range(nb):
        r0 = i * args.chunk
        r1 = min(r0 + args.chunk, H)
        lm = land[r0:r1]
        if args.no_candidate:
            sel = lm
        else:
            sel = lm & candidate[r0:r1]
        if not sel.any():
            continue
        block = st.read(MT_BANDS, window=((r0, r1), (0, W)))  # (T, rows, W)
        T = block.shape[0]
        rr, cc = np.where(sel)
        vh = block[:, rr, cc].T.astype(np.float32)            # (Npix, T)
        good = ~np.any(vh == nodata, axis=1) if nodata is not None else np.ones(len(vh), bool)
        vh = vh[good]
        rr, cc = rr[good], cc[good]
        if len(vh) == 0:
            continue
        vh = interp_bad_cols(vh, bad_cols)
        n = count_cycles_vec(smooth_clip(vh))
        ci_out[r0 + rr, cc] = np.clip(n, 0, 3).astype(np.uint8)
        if (i + 1) % 5 == 0 or i == nb - 1:
            print(f"  chunk {i+1}/{nb} (rows {r0}-{r1})")

    paddy = (ci_out >= 1).astype(np.uint8)

    os.makedirs(args.output, exist_ok=True)
    prof = st.profile.copy()
    prof.update(count=1, dtype='uint8', nodata=0, compress='lzw', tiled=True)
    with rasterio.open(f"{args.output}/cropping_intensity.tif", 'w', **prof) as d:
        d.write(ci_out, 1)
    with rasterio.open(f"{args.output}/paddy_mask.tif", 'w', **prof) as d:
        d.write(paddy, 1)

    # stats
    tr = st.transform
    lat_c = tr.f + tr.e * (H / 2.0)
    px_ha = abs(tr.a) * 111320.0 * math.cos(math.radians(lat_c)) * abs(tr.e) * 111320.0 / 10000.0
    npad = int(paddy.sum())
    with open(f"{args.output}/statistics.txt", 'w') as f:
        f.write("Cycle-based Paddy Mask + Cropping Intensity (MT 2024/25)\n")
        f.write("="*60 + "\n")
        cand_label = (args.candidate_mask if args.candidate_mask
                      else ('none' if args.no_candidate else 'Model-B union'))
        f.write(f"min_rise={MIN_RISE} dBx100, bands {MT_BANDS[0]}-{MT_BANDS[-1]}, "
                f"candidate={cand_label}\n")
        f.write(f"anomalous ~0 dB bands interpolated: {bad_bands} {bad_names}\n")
        f.write(f"pixel area: {px_ha:.4f} ha\n\n")
        f.write(f"Paddy pixels: {npad:,}\n")
        f.write(f"Paddy area: {npad*px_ha:,.0f} ha\n\n")
        f.write("Cropping intensity distribution (within paddy):\n")
        for c in (1, 2, 3):
            n = int((ci_out == c).sum())
            f.write(f"  {c}{'+' if c==3 else ''} crop(s): {n:,} px = {n*px_ha:,.0f} ha "
                    f"({100*n/npad:.1f}% of paddy)\n")
        if npad:
            mean_ci = float((ci_out[paddy == 1]).mean())
            f.write(f"\nAggregate Cropping Index (mean cycles over paddy): {mean_ci:.2f}\n")
    print(open(f"{args.output}/statistics.txt").read())


if __name__ == '__main__':
    main()
