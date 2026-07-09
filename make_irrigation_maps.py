#!/usr/bin/env python3
"""
Spatial per-tertiary-block (petak) maps of irrigation performance for DI Klambu:
3 choropleth panels — mean Satisfaction Index (SI), Christiansen Uniformity (CU),
and Reliability (RI) — joining klambu.gpkg petak geometry to the block-result CSVs.
Fills the 'no maps' gap for Sasaran 2 (which previously had only time-series/histograms).
"""
import csv, numpy as np
from osgeo import ogr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

GPKG='2026/irrigation_performance/klambu.gpkg'
UNI='results_csv/uniformity_block_results.csv'
REL='results_csv/reliability_block_results.csv'
OUT='2026/figures/fig_irrigation_klambu_maps.png'

def load(csvf, key):
    d={}
    with open(csvf) as f:
        for r in csv.DictReader(f):
            try: d[str(int(float(r['norec'])))]=float(r[key])
            except: pass
    return d

def ring_xy(geom):
    """exterior ring(s) of (Multi)Polygon -> list of Nx2 arrays (in km, local origin)."""
    polys=[]
    gt=geom.GetGeometryName()
    geoms=[geom.GetGeometryRef(i) for i in range(geom.GetGeometryCount())] if gt=='MULTIPOLYGON' else [geom]
    for g in geoms:
        if g is None: continue
        ring=g.GetGeometryRef(0) if g.GetGeometryName()=='POLYGON' else g
        pts=ring.GetPoints()
        if pts: polys.append(np.array([(p[0],p[1]) for p in pts]))
    return polys

def main():
    si=load(UNI,'mean_SI'); cu=load(UNI,'uniformity'); ri=load(REL,'reliability')
    ds=ogr.Open(GPKG); lyr=ds.GetLayer()
    feats=[]
    for ft in lyr:
        nr=str(ft.GetField('norec')); g=ft.GetGeometryRef()
        if g is None: continue
        feats.append((nr, ring_xy(g.Clone())))
    # local origin (km) for readable axes
    allx=np.concatenate([p[:,0] for _,ps in feats for p in ps]); ally=np.concatenate([p[:,1] for _,ps in feats for p in ps])
    x0,y0=allx.min(),ally.min()
    panels=[('Satisfaction Index (SI)',si,'RdYlGn',(0.4,1.0)),
            ('Christiansen Uniformity (CU)',cu,'RdYlGn',(0.6,1.0)),
            ('Reliability (RI)',ri,'RdYlGn',(0.5,1.0))]
    fig,axes=plt.subplots(1,3,figsize=(16,5.6))
    for ax,(title,vals,cmap,(vmin,vmax)) in zip(axes,panels):
        patches=[]; colors=[]
        cm=plt.get_cmap(cmap)
        for nr,ps in feats:
            v=vals.get(nr,None)
            for p in ps:
                patches.append(MplPoly((p[:,:2]-[x0,y0])/1000.0, closed=True))
                colors.append(cm((np.clip(v,vmin,vmax)-vmin)/(vmax-vmin)) if v is not None else (0.85,0.85,0.85,1))
        pc=PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=0.1)
        ax.add_collection(pc); ax.autoscale_view(); ax.set_aspect('equal')
        ax.set_title(f'{title}\nDI Klambu (per petak tersier)', fontsize=11)
        ax.set_xlabel('km'); ax.set_ylabel('km')
        sm=plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin,vmax)); sm.set_array([])
        plt.colorbar(sm,ax=ax,shrink=0.7,label=title.split('(')[0].strip())
        nrep=sum(1 for nr,_ in feats if vals.get(nr) is not None)
        ax.text(0.02,0.98,f'n={nrep} petak\nmean={np.mean([v for v in vals.values()]):.2f}',
                transform=ax.transAxes,va='top',fontsize=8,bbox=dict(boxstyle='round',fc='white',alpha=0.8))
    fig.suptitle('Peta Kinerja Irigasi per Petak Tersier — DI Klambu (MT 2023/24, berbasis Sentinel-1)',fontsize=12)
    plt.tight_layout(); plt.savefig(OUT,dpi=150,bbox_inches='tight'); plt.close()
    print(f'wrote {OUT}  (SI n={len(si)}, CU n={len(cu)}, RI n={len(ri)})')

if __name__=='__main__':
    main()
