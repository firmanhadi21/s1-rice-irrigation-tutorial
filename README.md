# Sentinel-1 Rice & Irrigation Monitoring — Hands-on Tutorial

A reproducible, laptop-friendly tutorial that turns free **Sentinel-1 radar** time series into three
operational products for water-resources management over Java, Indonesia:

- 🌾 **Paddy-field map** — VH phenology + SMOTE-balanced classifier + multi-year consensus
- 📅 **Planting index** — data-driven cropping-cycle counting (1× / 2× / 3×)
- 💧 **Irrigation performance** — Satisfaction (SI), Uniformity (CU), Reliability (RI) per tertiary block

## ▶️ Read it online

**https://firmanhadi21.github.io/s1-rice-irrigation-tutorial/**

## Run it yourself

Everything runs on a laptop (CPU only). Clone this repo, create the environment, fetch a small
Sentinel-1 area of interest from Google Earth Engine (free account), and follow the chapters.

```bash
git clone https://github.com/firmanhadi21/s1-rice-irrigation-tutorial.git
cd s1-rice-irrigation-tutorial
conda create -n s1rice python=3.10 -y && conda activate s1rice
pip install tensorflow rasterio geopandas rioxarray scikit-learn imbalanced-learn \
            numpy pandas matplotlib scipy joblib earthengine-api requests
```

Then open the [**Setup**](https://firmanhadi21.github.io/s1-rice-irrigation-tutorial/setup.html)
chapter. A trained model, DI Klambu tertiary-block boundaries, and reference outputs ship with this
repo; the Sentinel-1 stack for the sample area is fetched from Earth Engine.

## What's here

- `tutorial/` — Quarto source of the site (`.qmd`, images, `scripts/`)
- core scripts: `train_paddy_vh_smote.py`, `predict_paddy_vh_multiyear.py`,
  `create_multiyear_composite_paddy.py`, `cropping_intensity.py`, `produce_kc_map.py`,
  `calculate_irrigation_indices.py`, `make_irrigation_maps.py` (+ `utils_paddy_vh.py`, `config_paddy_vh.py`)
- `model_files_paddy_vh/` — trained paddy model + scaler + training features
- `2026/irrigation_performance/` — DI Klambu boundaries + reference SI/CU/RI results
- `tutorial/scripts/test_tutorial.sh` — smoke-test that runs every step (verified end to end)

## Notes

- No GPU required. The full Java-scale products in the source study were produced on an HPC from
  ~1,000 Sentinel-1 scenes; here you reproduce the **same methods** on a small sample AOI.
- The more accurate growth-phase mapping uses open-source Sentinel-1/2 fusion (MOGPR / FuseTS) — see
  the "Going further" chapter.

## License

MIT — see `LICENSE`. Sentinel-1 data © ESA/Copernicus.
