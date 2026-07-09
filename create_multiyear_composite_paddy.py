#!/usr/bin/env python3
"""
Multi-Year Composite Paddy Map

Creates a high-confidence paddy map by combining predictions from multiple years.
A pixel is marked as paddy only if it's detected as paddy in BOTH years,
ensuring we only map areas that are consistently paddy fields.

Usage:
    python create_multiyear_composite_paddy.py \
        --years 2023 2024 \
        --periods 2023:"15 25" \
        --periods 2024:"15 22" \
        --output composite_paddy_2023_2024
"""

import argparse
import numpy as np
import rasterio
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_predictions_for_year(year, periods, base_dir='predictions_paddy'):
    """
    Load all prediction and confidence maps for a given year

    Returns:
        paddy_maps: list of arrays (1 per period)
        confidence_maps: list of arrays (1 per period)
        profile: rasterio profile from first file
    """
    paddy_maps = []
    confidence_maps = []
    profile = None

    logger.info(f"\nLoading predictions for {year}:")
    for period in periods:
        pred_dir = Path(base_dir) / str(year) / f'period_{period:02d}'
        pred_file = pred_dir / 'paddy_predictions.tif'
        conf_file = pred_dir / 'confidence.tif'

        if not pred_file.exists():
            logger.warning(f"  ✗ Missing: {pred_file}")
            continue

        with rasterio.open(pred_file) as src:
            paddy = src.read(1)
            if profile is None:
                profile = src.profile
            logger.info(f"  ✓ Period {period}: {pred_file}")

        with rasterio.open(conf_file) as src:
            confidence = src.read(1)

        paddy_maps.append(paddy)
        confidence_maps.append(confidence)

    return paddy_maps, confidence_maps, profile


def create_year_composite(paddy_maps, confidence_maps, min_detections=2,
                         min_confidence=0.7, min_mean_confidence=0.5):
    """
    Create composite for a single year with confidence filtering:
    - Paddy: 1 if paddy detected in >= min_detections periods AND meets confidence criteria
    - Max_confidence: highest confidence across all periods
    - Frequency: number of periods with paddy detection

    Args:
        min_detections: Minimum number of periods required for paddy classification (default: 2)
        min_confidence: Minimum confidence threshold for any detection (default: 0.0 = no filtering)
        min_mean_confidence: Minimum mean confidence across detections (default: 0.0 = no filtering)
    """
    if not paddy_maps:
        return None, None, None

    # Stack all periods
    paddy_stack = np.stack(paddy_maps, axis=0)  # (n_periods, height, width)
    conf_stack = np.stack(confidence_maps, axis=0)

    # Apply minimum confidence filter to detections
    if min_confidence > 0:
        # Only count detections where confidence >= threshold
        high_conf_detections = (paddy_stack == 1) & (conf_stack >= min_confidence)
    else:
        high_conf_detections = (paddy_stack == 1)

    # Frequency: how many high-confidence periods detected as paddy
    frequency = np.sum(high_conf_detections, axis=0).astype(np.uint8)

    # Stricter criteria: paddy if detected in >= min_detections periods
    # This reduces false positives from single-period detections
    year_paddy = (frequency >= min_detections).astype(np.uint8)

    # Calculate mean confidence for pixels detected as paddy
    if min_mean_confidence > 0:
        # Get mean confidence only where paddy was detected
        mean_conf = np.zeros_like(year_paddy, dtype=np.float32)
        for i in range(conf_stack.shape[0]):
            mean_conf += conf_stack[i] * high_conf_detections[i]

        # Avoid division by zero
        mean_conf = np.where(frequency > 0, mean_conf / frequency, 0)

        # Apply mean confidence threshold
        year_paddy = year_paddy & (mean_conf >= min_mean_confidence)

    # Maximum confidence across all periods
    max_confidence = np.max(conf_stack, axis=0)

    return year_paddy, max_confidence, frequency


def create_multiyear_composite(year_composites, year_confidences, min_years=2):
    """
    Create multi-year consensus map:
    - Only mark as paddy if detected in >= min_years
    - Use maximum confidence across all years

    Args:
        year_composites: dict {year: ever_paddy_map}
        year_confidences: dict {year: max_confidence_map}
        min_years: minimum number of years for consensus (default: 2 = both years)

    Returns:
        consensus_paddy: 1 where paddy in >= min_years, 0 otherwise
        consensus_confidence: maximum confidence across all years
        num_years_detected: how many years detected as paddy (0-N)
    """
    years = sorted(year_composites.keys())

    # Stack year composites
    year_stack = np.stack([year_composites[y] for y in years], axis=0)
    conf_stack = np.stack([year_confidences[y] for y in years], axis=0)

    # Count how many years detected as paddy
    num_years_detected = np.sum(year_stack == 1, axis=0).astype(np.uint8)

    # Consensus: paddy if detected in >= min_years
    consensus_paddy = (num_years_detected >= min_years).astype(np.uint8)

    # Maximum confidence across all years
    consensus_confidence = np.max(conf_stack, axis=0)

    return consensus_paddy, consensus_confidence, num_years_detected


def save_outputs(consensus_paddy, consensus_confidence, num_years_detected,
                 profile, output_dir, years):
    """Save all outputs with statistics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update profile for output - use 0 as nodata for uint8
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0)

    # 1. Save consensus paddy map
    consensus_file = output_dir / 'consensus_paddy_map.tif'
    with rasterio.open(consensus_file, 'w', **profile) as dst:
        dst.write(consensus_paddy, 1)
    logger.info(f"✓ Saved: {consensus_file}")

    # 2. Save number of years detected
    years_file = output_dir / 'num_years_detected.tif'
    with rasterio.open(years_file, 'w', **profile) as dst:
        dst.write(num_years_detected, 1)
    logger.info(f"✓ Saved: {years_file}")

    # 3. Save consensus confidence
    profile.update(dtype=rasterio.float32, nodata=-9999.0)
    conf_file = output_dir / 'consensus_confidence.tif'
    with rasterio.open(conf_file, 'w', **profile) as dst:
        dst.write(consensus_confidence.astype(np.float32), 1)
    logger.info(f"✓ Saved: {conf_file}")

    # 4. Statistics
    total_pixels = np.sum(consensus_paddy >= 0)  # all valid pixels
    paddy_pixels = np.sum(consensus_paddy == 1)
    # Pixel area from the actual geotransform. Stack is EPSG:4326 (degrees);
    # convert with the raster center latitude (~0.2479 ha/pixel at Java ~7S,
    # i.e. 49.6 m x 50 m). Avoids the flat-0.25 (nominal 50 m) ~0.8% overestimate.
    import math
    tr = profile['transform']
    crs = profile.get('crs')
    if crs is not None and crs.is_geographic:
        lat_c = tr.f + tr.e * (consensus_paddy.shape[0] / 2.0)
        px_area_ha = abs(tr.a) * 111320.0 * math.cos(math.radians(lat_c)) * abs(tr.e) * 111320.0 / 10000.0
    else:
        px_area_ha = abs(tr.a) * abs(tr.e) / 10000.0
    paddy_area_ha = paddy_pixels * px_area_ha

    stats_file = output_dir / 'statistics.txt'
    with open(stats_file, 'w') as f:
        f.write(f"Multi-Year Consensus Paddy Map\n")
        f.write(f"Years: {', '.join(map(str, years))}\n")
        f.write(f"Consensus Criteria: Paddy detected in ALL {len(years)} years\n")
        f.write(f"="*60 + "\n\n")

        f.write(f"Total valid pixels: {total_pixels:,}\n\n")

        f.write(f"Consensus Paddy Detection:\n")
        f.write(f"-"*60 + "\n")
        f.write(f"Paddy pixels: {paddy_pixels:,}\n")
        f.write(f"Paddy percentage: {100*paddy_pixels/total_pixels:.2f}%\n")
        f.write(f"Estimated area: {paddy_area_ha:,.2f} hectares\n\n")

        f.write(f"Detection by Number of Years:\n")
        f.write(f"-"*60 + "\n")
        for n in range(len(years) + 1):
            n_pixels = np.sum(num_years_detected == n)
            pct = 100 * n_pixels / total_pixels
            f.write(f"  {n} year(s): {n_pixels:,} pixels ({pct:.2f}%)\n")

        f.write(f"\nConfidence Statistics:\n")
        f.write(f"-"*60 + "\n")
        paddy_conf = consensus_confidence[consensus_paddy == 1]
        if len(paddy_conf) > 0:
            f.write(f"  Mean: {np.mean(paddy_conf):.3f}\n")
            f.write(f"  Median: {np.median(paddy_conf):.3f}\n")
            f.write(f"  Std: {np.std(paddy_conf):.3f}\n")
            f.write(f"  Min: {np.min(paddy_conf):.3f}\n")
            f.write(f"  Max: {np.max(paddy_conf):.3f}\n")

    logger.info(f"✓ Saved: {stats_file}")

    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Consensus paddy map
    ax = axes[0]
    cmap = ListedColormap(['white', 'darkgreen'])
    im1 = ax.imshow(consensus_paddy, cmap=cmap, interpolation='nearest')
    ax.set_title(f'Consensus Paddy Map\n(Detected in ALL {len(years)} years)')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, label='0=Non-Paddy, 1=Paddy', shrink=0.6)

    # Number of years detected
    ax = axes[1]
    cmap = plt.cm.YlGnBu
    im2 = ax.imshow(num_years_detected, cmap=cmap, interpolation='nearest',
                    vmin=0, vmax=len(years))
    ax.set_title(f'Number of Years Detected as Paddy\n(0 to {len(years)})')
    ax.axis('off')
    plt.colorbar(im2, ax=ax, label='Number of Years', shrink=0.6)

    # Confidence
    ax = axes[2]
    im3 = ax.imshow(consensus_confidence, cmap='RdYlGn', interpolation='nearest',
                    vmin=0, vmax=1)
    ax.set_title('Maximum Confidence\n(Across all years)')
    ax.axis('off')
    plt.colorbar(im3, ax=ax, label='Confidence', shrink=0.6)

    plt.tight_layout()
    viz_file = output_dir / 'composite_visualization.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {viz_file}")

    # Print statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"STATISTICS")
    logger.info(f"{'='*60}")
    logger.info(f"Consensus paddy pixels: {paddy_pixels:,} ({100*paddy_pixels/total_pixels:.2f}%)")
    logger.info(f"Estimated area: {paddy_area_ha:,.2f} hectares")


def main():
    parser = argparse.ArgumentParser(
        description='Create multi-year consensus paddy map'
    )
    parser.add_argument('--years', nargs='+', type=int, required=True,
                        help='Years to combine (e.g., 2023 2024)')
    parser.add_argument('--periods', action='append', required=True, metavar='YEAR:PERIODS',
                        help='Per-year periods as YEAR:"p1 p2 ...". Repeatable. '
                             'E.g. --periods 2024:"7 9 11" --periods 2025:"7 9 11" --periods 2026:"7 9"')
    parser.add_argument('--predictions-dir', default='predictions_paddy',
                        help='Base directory for predictions')
    parser.add_argument('--output', required=True,
                        help='Output directory name')
    parser.add_argument('--min-years', type=int, default=2,
                        help='Minimum years for consensus (default: 2 = all years)')
    parser.add_argument('--min-detections', type=int, default=2,
                        help='Minimum period detections per year (default: 2, reduces false positives)')
    parser.add_argument('--min-confidence', type=float, default=0.0,
                        help='Minimum confidence for each detection (0.0-1.0, default: 0.0 = no filter)')
    parser.add_argument('--min-mean-confidence', type=float, default=0.0,
                        help='Minimum mean confidence across detections (0.0-1.0, default: 0.0 = no filter)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("MULTI-YEAR CONSENSUS PADDY MAPPING")
    logger.info("="*80)
    logger.info(f"Years: {args.years}")
    logger.info(f"\nFiltering criteria:")
    logger.info(f"  - Within-year: Paddy in >= {args.min_detections} periods")
    logger.info(f"  - Multi-year: Detected in >= {args.min_years} years")
    if args.min_confidence > 0:
        logger.info(f"  - Min confidence per detection: {args.min_confidence:.2f}")
    if args.min_mean_confidence > 0:
        logger.info(f"  - Min mean confidence: {args.min_mean_confidence:.2f}")

    # Parse per-year periods from repeated --periods YEAR:"p1 p2 ..."
    periods_map = {}
    for entry in args.periods:
        if ':' not in entry:
            parser.error(f"--periods must be YEAR:PERIODS, got '{entry}'")
        ystr, pstr = entry.split(':', 1)
        periods_map[int(ystr)] = [int(p) for p in pstr.replace(',', ' ').split()]

    year_composites = {}
    year_confidences = {}
    year_frequencies = {}
    profile = None

    # Process each year
    for year in args.years:
        if year not in periods_map:
            logger.error(f"No --periods provided for year {year}")
            continue
        periods = periods_map[year]

        # Load predictions for this year
        paddy_maps, conf_maps, prof = load_predictions_for_year(
            year, periods, args.predictions_dir
        )

        if not paddy_maps:
            logger.error(f"No predictions found for {year}")
            continue

        if profile is None:
            profile = prof

        # Create year composite with all filtering criteria
        year_paddy, max_conf, freq = create_year_composite(
            paddy_maps, conf_maps,
            min_detections=args.min_detections,
            min_confidence=args.min_confidence,
            min_mean_confidence=args.min_mean_confidence
        )

        year_composites[year] = year_paddy
        year_confidences[year] = max_conf
        year_frequencies[year] = freq

        # Log year statistics
        n_paddy = np.sum(year_paddy == 1)
        total = year_paddy.size
        logger.info(f"\n{year} Summary:")
        logger.info(f"  Paddy detected (>= {args.min_detections} periods): {n_paddy:,} pixels ({100*n_paddy/total:.2f}%)")
        logger.info(f"  Max detections in any pixel: {np.max(freq)}")
        logger.info(f"  Mean confidence: {np.mean(max_conf):.3f}")

    # Create multi-year consensus
    logger.info(f"\n{'='*80}")
    logger.info(f"CREATING MULTI-YEAR CONSENSUS")
    logger.info(f"{'='*80}")

    consensus_paddy, consensus_conf, num_years = create_multiyear_composite(
        year_composites, year_confidences, args.min_years
    )

    # Save outputs
    save_outputs(consensus_paddy, consensus_conf, num_years,
                 profile, args.output, args.years)

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {args.output}")


if __name__ == '__main__':
    main()
