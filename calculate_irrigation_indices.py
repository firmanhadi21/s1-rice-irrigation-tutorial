#!/usr/bin/env python3
"""
Calculate Irrigation Performance Indices for Daerah Irigasi (Irrigation District)

Computes three key irrigation performance indices from satellite-derived paddy maps:
1. Satisfaction Index (SI): fraction of paddy area actively growing per period
2. Uniformity Index (CU): Christiansen's Uniformity Coefficient across zones
3. Reliability Index (RI): fraction of periods with SI above threshold per zone

Usage:
    # With tertiary block boundaries
    python calculate_irrigation_indices.py \
        --predictions-dir predictions_di/DI_Rentang/2023 \
        --periods 8 10 12 14 16 18 20 \
        --di-boundary data/di/DI_Rentang.gpkg \
        --tertiary-blocks data/di/DI_Rentang_petak.gpkg \
        --output-dir irrigation_indices/DI_Rentang/2023

    # DI-level only (no tertiary blocks)
    python calculate_irrigation_indices.py \
        --predictions-dir predictions_di/DI_Rentang/2023 \
        --periods 8 10 12 14 16 18 20 \
        --di-boundary data/di/DI_Rentang.gpkg \
        --output-dir irrigation_indices/DI_Rentang/2023
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_predictions(predictions_dir, periods):
    """Load paddy predictions and confidences for specified periods."""
    preds = {}
    confs = {}
    profile = None

    for period in periods:
        period_dir = Path(predictions_dir) / f'period_{period:02d}'
        pred_file = period_dir / 'paddy_predictions.tif'
        conf_file = period_dir / 'confidence.tif'

        if not pred_file.exists():
            logger.warning(f"Missing prediction for period {period}: {pred_file}")
            continue

        with rasterio.open(pred_file) as src:
            preds[period] = src.read(1)
            if profile is None:
                profile = src.profile.copy()

        if conf_file.exists():
            with rasterio.open(conf_file) as src:
                confs[period] = src.read(1)

    logger.info(f"Loaded predictions for {len(preds)} periods")
    return preds, confs, profile


def rasterize_zones(gdf, profile, id_field=None):
    """
    Rasterize zone polygons to match prediction grid.

    Returns:
        zone_raster: (height, width) array with zone IDs (1-based)
        zone_names: dict mapping zone_id -> name
    """
    height = profile['height']
    width = profile['width']
    transform = profile['transform']

    # Reproject if needed
    crs = profile.get('crs')
    if crs and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Assign zone IDs
    if id_field and id_field in gdf.columns:
        # Use existing field
        zone_names = {}
        shapes = []
        for idx, row in gdf.iterrows():
            zid = idx + 1
            zone_names[zid] = str(row[id_field])
            shapes.append((row.geometry, zid))
    else:
        zone_names = {i + 1: f'Zone_{i + 1}' for i in range(len(gdf))}
        shapes = [(row.geometry, i + 1) for i, (_, row) in enumerate(gdf.iterrows())]

    zone_raster = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32
    )

    return zone_raster, zone_names


def compute_satisfaction_index(predictions, confidences, zone_raster, zone_ids, confidence_threshold=0.7):
    """
    Compute Satisfaction Index per zone per period.

    SI = (active paddy pixels with sufficient confidence) / (total paddy pixels in zone) * 100
    """
    # Build reference: total paddy pixels per zone (union across all periods)
    all_preds = np.stack(list(predictions.values()), axis=0)
    ever_paddy = np.any(all_preds == 1, axis=0)

    results = []
    periods = sorted(predictions.keys())

    for zone_id in zone_ids:
        zone_mask = zone_raster == zone_id
        total_paddy = np.sum(ever_paddy & zone_mask)

        if total_paddy == 0:
            for period in periods:
                results.append({'zone_id': zone_id, 'period': period, 'si': 0.0,
                                'active_paddy': 0, 'total_paddy': 0})
            continue

        for period in periods:
            pred = predictions[period]
            active = pred == 1

            # Apply confidence filter if available
            if period in confidences:
                conf = confidences[period]
                active = active & (conf >= confidence_threshold)

            active_paddy = np.sum(active & zone_mask)
            si = 100.0 * active_paddy / total_paddy

            results.append({
                'zone_id': zone_id,
                'period': period,
                'si': si,
                'active_paddy': int(active_paddy),
                'total_paddy': int(total_paddy)
            })

    return pd.DataFrame(results)


def compute_uniformity_index(si_df, zone_ids):
    """
    Compute Christiansen's Uniformity Coefficient per period.

    CU = 1 - (sum|xi - x_mean|) / (n * x_mean)
    """
    results = []
    periods = sorted(si_df['period'].unique())

    for period in periods:
        period_data = si_df[si_df['period'] == period]
        si_values = period_data['si'].values

        n = len(si_values)
        si_mean = np.mean(si_values)

        if si_mean > 0 and n > 1:
            cu = 1.0 - np.sum(np.abs(si_values - si_mean)) / (n * si_mean)
        else:
            cu = 0.0

        results.append({
            'period': period,
            'cu': cu,
            'si_mean': si_mean,
            'si_std': np.std(si_values),
            'si_min': np.min(si_values),
            'si_max': np.max(si_values),
            'n_zones': n
        })

    return pd.DataFrame(results)


def compute_reliability_index(si_df, zone_ids, si_threshold=50.0):
    """
    Compute Reliability Index per zone.

    RI = (number of periods where SI > threshold) / (total periods)
    """
    results = []
    n_periods = len(si_df['period'].unique())

    for zone_id in zone_ids:
        zone_data = si_df[si_df['zone_id'] == zone_id]
        n_above = np.sum(zone_data['si'] > si_threshold)
        ri = n_above / n_periods if n_periods > 0 else 0.0

        results.append({
            'zone_id': zone_id,
            'ri': ri,
            'periods_above_threshold': int(n_above),
            'total_periods': n_periods,
            'mean_si': zone_data['si'].mean(),
            'si_threshold': si_threshold
        })

    return pd.DataFrame(results)


def plot_si_timeseries(si_df, zone_names, output_file):
    """Plot SI over time per zone."""
    fig, ax = plt.subplots(figsize=(12, 6))

    zones = sorted(si_df['zone_id'].unique())
    if len(zones) <= 10:
        for zone_id in zones:
            zone_data = si_df[si_df['zone_id'] == zone_id].sort_values('period')
            label = zone_names.get(zone_id, f'Zone {zone_id}')
            ax.plot(zone_data['period'], zone_data['si'], 'o-', label=label, markersize=4)
        ax.legend(fontsize=8, ncol=2)
    else:
        # Too many zones - plot mean + std band
        grouped = si_df.groupby('period')['si'].agg(['mean', 'std']).reset_index()
        ax.plot(grouped['period'], grouped['mean'], 'b-o', label='Mean SI', markersize=5)
        ax.fill_between(grouped['period'],
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2, color='blue', label='Std band')
        ax.legend()

    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Satisfaction Index (%)', fontsize=12)
    ax.set_title('Satisfaction Index Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_uniformity_timeseries(cu_df, output_file):
    """Plot CU over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cu_df['period'], cu_df['cu'], 'g-o', markersize=6, linewidth=2)
    ax.axhline(y=0.85, color='r', linestyle='--', alpha=0.7, label='Target CU = 0.85')
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Uniformity Coefficient', fontsize=12)
    ax.set_title("Christiansen's Uniformity Coefficient Over Time", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Calculate irrigation performance indices')
    parser.add_argument('--predictions-dir', required=True, help='Directory with per-period predictions')
    parser.add_argument('--periods', type=int, nargs='+', required=True, help='Period numbers')
    parser.add_argument('--di-boundary', required=True, help='DI boundary shapefile/GeoPackage')
    parser.add_argument('--tertiary-blocks', help='Optional tertiary block boundaries')
    parser.add_argument('--zone-id-field', help='Field name for zone ID in tertiary blocks')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--si-threshold', type=float, default=50.0,
                        help='SI threshold for reliability calculation (default: 50%%)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Minimum confidence for active paddy (default: 0.7)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    predictions, confidences, profile = load_predictions(args.predictions_dir, args.periods)
    if not predictions:
        logger.error("No predictions loaded. Exiting.")
        sys.exit(1)

    # Load and rasterize zones
    if args.tertiary_blocks and os.path.exists(args.tertiary_blocks):
        logger.info(f"Using tertiary block zones: {args.tertiary_blocks}")
        gdf = gpd.read_file(args.tertiary_blocks)
        zone_raster, zone_names = rasterize_zones(gdf, profile, args.zone_id_field)
    else:
        logger.info("No tertiary blocks - computing DI-level indices only")
        gdf = gpd.read_file(args.di_boundary)
        zone_raster, zone_names = rasterize_zones(gdf, profile)

    zone_ids = sorted(zone_names.keys())
    logger.info(f"Zones: {len(zone_ids)}")

    # Compute indices
    logger.info("Computing Satisfaction Index...")
    si_df = compute_satisfaction_index(predictions, confidences, zone_raster, zone_ids,
                                       args.confidence_threshold)

    logger.info("Computing Uniformity Index...")
    cu_df = compute_uniformity_index(si_df, zone_ids)

    logger.info("Computing Reliability Index...")
    ri_df = compute_reliability_index(si_df, zone_ids, args.si_threshold)

    # Save CSVs
    si_df.to_csv(output_dir / 'satisfaction_index.csv', index=False)
    cu_df.to_csv(output_dir / 'uniformity_index.csv', index=False)
    ri_df.to_csv(output_dir / 'reliability_index.csv', index=False)

    # Summary statistics
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write("Irrigation Performance Indices Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"DI Boundary: {args.di_boundary}\n")
        f.write(f"Periods analyzed: {sorted(predictions.keys())}\n")
        f.write(f"Number of zones: {len(zone_ids)}\n")
        f.write(f"Confidence threshold: {args.confidence_threshold}\n")
        f.write(f"SI threshold for reliability: {args.si_threshold}%\n\n")

        f.write("Satisfaction Index (SI) Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Overall mean SI: {si_df['si'].mean():.1f}%\n")
        f.write(f"  SI range: {si_df['si'].min():.1f}% - {si_df['si'].max():.1f}%\n\n")

        f.write("Uniformity Index (CU) Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Mean CU: {cu_df['cu'].mean():.3f}\n")
        f.write(f"  CU range: {cu_df['cu'].min():.3f} - {cu_df['cu'].max():.3f}\n")
        f.write(f"  Periods with CU > 0.85: {np.sum(cu_df['cu'] > 0.85)} / {len(cu_df)}\n\n")

        f.write("Reliability Index (RI) Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Mean RI: {ri_df['ri'].mean():.3f}\n")
        f.write(f"  RI range: {ri_df['ri'].min():.3f} - {ri_df['ri'].max():.3f}\n")
        f.write(f"  Zones with RI > 0.75: {np.sum(ri_df['ri'] > 0.75)} / {len(ri_df)}\n")

    # Plots
    plot_si_timeseries(si_df, zone_names, output_dir / 'si_timeseries.png')
    plot_uniformity_timeseries(cu_df, output_dir / 'uniformity_timeseries.png')

    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
