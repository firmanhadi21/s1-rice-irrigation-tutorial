#!/usr/bin/env python3
"""Fetch a small Sentinel-1 VH 12-day composite stack for the tutorial AOI (DI Klambu).

Builds a multi-band GeoTIFF (one band per 12-day period) using Google Earth Engine, so the rest
of the tutorial can run on a laptop without the full Java-scale stack. Output band naming matches
what the project scripts expect: <year>_Period_<n>.

Prerequisites
-------------
  pip install earthengine-api requests
  # Auth, either:
  earthengine authenticate                    # interactive (free account), OR
  #  ... just place a service-account key `ee-geodetic.json` in the repo root
  #      (auto-detected) or pass --service-account-key /path/to/key.json  -> no
  #      interactive login needed (ideal on an HPC).

Usage
-----
  python fetch_sample_aoi.py --year 2024 --out ../../data/sample/klambu_vh_2024.tif
  # smaller/faster (fewer periods):
  python fetch_sample_aoi.py --year 2024 --start-period 7 --end-period 20 --out ...

Notes
-----
- AOI defaults to a ~22 x 18 km box over DI Klambu (Grobogan, Central Java). Override with --bbox.
- 50 m resolution, VH polarization, IW mode, ascending+descending merged, median per period.
- Small AOIs download directly via getDownloadURL; for larger AOIs use --to-drive.
"""
import argparse
import io
import zipfile
from pathlib import Path

import ee
import requests

# DI Klambu bounding box [west, south, east, north] (deg)
KLAMBU_BBOX = [110.72, -7.10, 110.94, -6.94]


def init_ee(project=None, sa_key=None):
    """Initialize Earth Engine, preferring a service-account key when available.

    Order: explicit --service-account-key -> auto-detected ee-geodetic.json (repo
    root or cwd) -> interactive/default credentials. Lets the fetch run head-less
    on an HPC without `earthengine authenticate`.
    """
    import json
    if sa_key is None:
        here = Path(__file__).resolve()
        for cand in (Path.cwd() / "ee-geodetic.json", here.parents[2] / "ee-geodetic.json"):
            if cand.exists():
                sa_key = str(cand)
                break
    if sa_key and Path(sa_key).exists():
        info = json.load(open(sa_key))
        email, proj = info["client_email"], project or info.get("project_id")
        ee.Initialize(ee.ServiceAccountCredentials(email, sa_key), project=proj)
        print(f"GEE: service account {email} (project {proj})")
    else:
        ee.Initialize(project=project) if project else ee.Initialize()
        print("GEE: interactive/default credentials")


def periods_12day(year):
    """Return list of (period_no, start 'YYYY-MM-DD', end) for 12-day periods in `year`."""
    import datetime as dt
    out, p, d0 = [], 1, dt.date(year, 1, 1)
    while d0.year == year:
        d1 = d0 + dt.timedelta(days=12)
        out.append((p, d0.isoformat(), d1.isoformat()))
        p, d0 = p + 1, d1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--start-period", type=int, default=1)
    ap.add_argument("--end-period", type=int, default=31)
    ap.add_argument("--bbox", type=float, nargs=4, default=KLAMBU_BBOX,
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--res", type=float, default=50.0)
    ap.add_argument("--out", default="../../data/sample/klambu_vh_2024.tif")
    ap.add_argument("--project", default=None, help="GEE Cloud project id (if required)")
    ap.add_argument("--service-account-key", default=None,
                    help="GEE service-account JSON (e.g. ee-geodetic.json). If omitted, "
                         "auto-detects ee-geodetic.json in the repo root/cwd, else uses "
                         "interactive credentials.")
    ap.add_argument("--to-drive", action="store_true", help="export to Google Drive instead")
    a = ap.parse_args()

    init_ee(a.project, a.service_account_key)
    aoi = ee.Geometry.Rectangle(a.bbox)

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select("VH"))

    bands = []
    for p, s, e in periods_12day(a.year):
        if p < a.start_period or p > a.end_period:
            continue
        comp = (s1.filterDate(s, e).median()
                .rename(f"{a.year}_Period_{p}")
                .clip(aoi))
        bands.append(comp)
    stack = ee.Image.cat(bands).toFloat()
    print(f"AOI {a.bbox} | year {a.year} | periods "
          f"{a.start_period}-{a.end_period} ({len(bands)} bands) | {a.res} m")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.to_drive:
        task = ee.batch.Export.image.toDrive(
            image=stack, description=out.stem, region=aoi, scale=a.res,
            fileFormat="GeoTIFF", maxPixels=1e9)
        task.start()
        print(f"Export to Drive started (task '{out.stem}'); download the GeoTIFF to {out}")
        return

    url = stack.getDownloadURL({"scale": a.res, "region": aoi,
                                "format": "GEO_TIFF", "filePerBand": False})
    print("downloading ...")
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    # GEE returns a single GeoTIFF, or a zip for some requests — handle both.
    if r.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            tif = [n for n in z.namelist() if n.endswith(".tif")][0]
            out.write_bytes(z.read(tif))
    else:
        out.write_bytes(r.content)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
