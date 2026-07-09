#!/usr/bin/env python3
"""
Kc (crop coefficient) map for MT 2024/25 from the trough-anchored growth-phase/DAT
estimator (S1-only). DAT = periods-since-transplant-trough x 12; Kc from the
continuous FAO-56 rice 110-day curve (per WATER_ADEQUACY_INDEX_METHODOLOGY.md §5):
  DAT 0-10: Kc=1.05 (initial) | 10-80: 1.05->1.20 | 80-110: 1.20->0.95 | >110: 0.
Intermediate product (crop water-demand coefficient) — error bounded by the
phase/DAT accuracy (~66% / 100%-within-1-phase, BulakBakal). NOT a performance metric.
"""
import numpy as np, rasterio, argparse
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

RG='/home/unika_sianturi/work/rice-growth-stage-mapping'
STACK=f'{RG}/stacks/java_vh_2024_2026_50m.tif'
PMASK='cropping_intensity_lbs_mt2024_25/paddy_mask.tif'
CLIP=(-3500.0,-100.0); WIN=10

def kc_from_dat(dat):
    kc=np.zeros_like(dat,dtype=np.float32)
    init=(dat>=0)&(dat<10); ramp=(dat>=10)&(dat<80); decl=(dat>=80)&(dat<110)
    kc[init]=1.05
    kc[ramp]=1.05+(1.20-1.05)*(dat[ramp]-10)/70.0
    kc[decl]=1.20-(1.20-0.95)*(dat[decl]-80)/30.0
    return kc  # >=110 stays 0 (harvest/fallow, no demand)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--period-band',type=int,default=35)
    ap.add_argument('--out',default='kc_mt2024_25')
    ap.add_argument('--stack',default=None,help='override VH stack path (default: hardcoded)')
    ap.add_argument('--paddy-mask',default=None,help='override paddy mask path')
    a=ap.parse_args()
    global STACK, PMASK
    if a.stack: STACK=a.stack
    if a.paddy_mask: PMASK=a.paddy_mask
    import os; os.makedirs(a.out,exist_ok=True)
    P=a.period_band; bands=list(range(P-WIN+1,P+1))
    s=rasterio.open(STACK); nod=s.nodata; H,W=s.height,s.width
    pm=rasterio.open(PMASK).read(1)>0; rr,cc=np.where(pm)
    vh=np.empty((len(rr),WIN),np.float32)
    for j,b in enumerate(bands): vh[:,j]=s.read(b)[rr,cc]
    good=~np.any(vh==nod,axis=1)
    vhs=savgol_filter(np.clip(vh[good],*CLIP),5,2,axis=1)
    ps=(WIN-1)-np.argmin(vhs,axis=1)          # periods since transplant trough
    dat=ps*12.0
    kc=kc_from_dat(dat)
    out=np.zeros((H,W),np.float32); out[rr[good],cc[good]]=kc
    prof=s.profile.copy(); prof.update(count=1,dtype='float32',nodata=0.0,compress='lzw',tiled=True)
    tif=f'{a.out}/kc_band{P}.tif'
    with rasterio.open(tif,'w',**prof) as d: d.write(out,1)
    kcp=kc[kc>0]
    print(f"paddy {len(rr):,} px; target band {P}")
    print(f"Kc (active-crop pixels, n={len(kcp):,}): mean={kcp.mean():.3f} median={np.median(kcp):.3f} "
          f"min={kcp.min():.2f} max={kcp.max():.2f}")
    print(f"  Kc=0 (harvest/fallow/no-demand): {int((kc==0).sum()):,} ({100*(kc==0).mean():.1f}%)")
    # figure
    dec=10
    od=rasterio.open(tif).read(1,out_shape=(H//dec,W//dec),resampling=Resampling.nearest)
    b=s.bounds; ext=[b.left,b.right,b.bottom,b.top]
    od=np.ma.masked_equal(od,0)
    plt.figure(figsize=(13,6))
    im=plt.imshow(od,extent=ext,cmap='YlGnBu',vmin=0.95,vmax=1.20,interpolation='nearest')
    plt.colorbar(im,label='Kc (koefisien tanaman)',shrink=0.7)
    plt.title('Peta Koefisien Tanaman (Kc) Padi Pulau Jawa — ~pertengahan Feb 2025 (MT 2024/25)\n'
              'dari estimator fase/DAT berbasis S1 (kurva FAO-56 110-hari) — produk antara, bukan metrik kinerja')
    plt.xlabel('Bujur (°)'); plt.ylabel('Lintang (°)')
    plt.tight_layout(); png='2026/figures/fig_kc_mt2024_25.png'
    plt.savefig(png,dpi=150,bbox_inches='tight'); plt.close()
    print('wrote',tif,'and',png)

if __name__=='__main__':
    main()
