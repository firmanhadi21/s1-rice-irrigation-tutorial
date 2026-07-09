#!/usr/bin/env python3
"""
VH-Only Binary Paddy Prediction for Multi-Year Stack

This script handles a single VH stack with multiple years of data.
For example: 54 bands = 31 bands (2023) + 23 bands (2024)

Usage:
    python predict_paddy_vh_multiyear.py --period 15 --year 2023 --year-start-band 1
    python predict_paddy_vh_multiyear.py --period 15 --year 2024 --year-start-band 32
"""

import os
import sys

# Device selection. Default is CPU-only (historically forced to avoid an XLA
# libdevice issue). Set PADDY_USE_GPU=1 to use the GPU instead — then the
# inherited CUDA_VISIBLE_DEVICES is respected (e.g. pin one GPU with
# `CUDA_VISIBLE_DEVICES=0`). XLA JIT stays disabled either way.
if os.environ.get('PADDY_USE_GPU', '0') != '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    # On GPU, some ops still trigger XLA JIT, which fails with "libdevice not
    # found" unless we point XLA at the CUDA data dir (the conda env holds
    # nvvm/libdevice/libdevice.10.bc). Derive it from the active env prefix.
    _cuda_dir = os.environ.get('CONDA_PREFIX') or sys.prefix
    os.environ.setdefault('XLA_FLAGS', f'--xla_gpu_cuda_data_dir={_cuda_dir}')
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'

import argparse
import numpy as np
import rasterio
import tensorflow as tf
tf.config.optimizer.set_jit(False)

# When using the GPU, restrict to a single device and enable memory growth so we
# only grab what the (tiny) MLP needs — safe on a shared, near-full GPU box.
if os.environ.get('PADDY_USE_GPU', '0') == '1':
    _gpus = tf.config.list_physical_devices('GPU')
    if _gpus:
        try:
            tf.config.set_visible_devices(_gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(_gpus[0], True)
            print(f"[paddy] Using GPU: {_gpus[0].name} (memory growth on)")
        except RuntimeError as _e:
            print(f"[paddy] GPU config failed ({_e}); continuing on default device")
    else:
        print("[paddy] PADDY_USE_GPU=1 but no GPU visible; running on CPU")
from tensorflow import keras
import joblib
import gc
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# Import configurations
from config_paddy_vh import (
    PATHS, FILES, MODEL_FILES, FEATURE_PARAMS, PREDICTION_PARAMS,
    NUM_CLASSES, CLASS_NAMES, CLASS_COLORS
)

# Import VH-only utilities
from utils_paddy_vh import (
    extract_vh_features_vectorized,
    get_feature_names,
    TemperatureLayer
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_year_band_offset(year, year_start_band):
    """
    Calculate band offset for a specific year

    Args:
        year: Year (2023 or 2024)
        year_start_band: Starting band for this year (1-based)

    Returns:
        Band offset (0-based)
    """
    return year_start_band - 1  # Convert to 0-based


def load_model_artifacts(model_dir=None):
    """Load trained model, scaler, and feature names.

    If model_dir is given, load paddy_vh_model.keras / scaler.joblib from there
    (used to select the RG+SG model). Otherwise fall back to config MODEL_FILES.
    """
    logger.info("Loading model artifacts...")

    if model_dir:
        model_path = os.path.join(model_dir, os.path.basename(MODEL_FILES['MODEL']))
        scaler_path = os.path.join(model_dir, os.path.basename(MODEL_FILES['SCALER']))
    else:
        model_path = MODEL_FILES['MODEL']
        scaler_path = MODEL_FILES['SCALER']

    # Load model with custom objects
    try:
        model = keras.models.load_model(
            model_path,
            custom_objects={'TemperatureLayer': TemperatureLayer}
        )
        logger.info(f"✓ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Retrying with compile=False...")
        model = keras.models.load_model(
            model_path,
            custom_objects={'TemperatureLayer': TemperatureLayer},
            compile=False
        )
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        logger.info("✓ Model loaded and recompiled")

    # Load scaler
    scaler = joblib.load(scaler_path)
    logger.info(f"✓ Scaler loaded from {scaler_path}")

    # Load feature names
    feature_names = get_feature_names(n_previous=FEATURE_PARAMS['N_PREVIOUS'])
    logger.info(f"✓ Feature names loaded ({len(feature_names)} features)")

    return model, scaler, feature_names


def create_prediction_map_multiyear(vh_tif, model, scaler, period, year, year_start_band,
                                     year_periods, mask_file=None, skip_test=False, threshold=0.5,
                                     rg_sg=False):
    """
    Create paddy prediction for a specific year and period from multi-year stack

    Args:
        vh_tif: Path to multi-year VH stack
        model: Trained model
        scaler: Fitted scaler
        period: Period number within the year (1-31)
        year: Year (2023 or 2024)
        year_start_band: Starting band for this year (1-based)
        year_periods: Number of periods in this year (e.g., 31 for 2023, 23 for 2024)
        mask_file: Optional mask file
        skip_test: Skip test prediction
        threshold: Classification threshold (default: 0.5)

    Returns:
        Tuple of (predictions, confidences, profile)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"MULTIYEAR STACK PREDICTION - {year} Period {period}")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Calculate band offset
    band_offset = get_year_band_offset(year, year_start_band)

    logger.info(f"Year: {year}")
    logger.info(f"Year starts at band: {year_start_band}")
    logger.info(f"Band offset: {band_offset}")

    # Open VH stack
    with rasterio.open(vh_tif) as src_vh:
        total_bands = src_vh.count
        logger.info(f"Total bands in stack: {total_bands}")

        # Read only the bands for this year
        # For a 54-band stack: bands 1-31 (2023), bands 32-54 (2024)
        max_period = year_periods  # Use the specified number of periods for this year

        if period > max_period:
            raise ValueError(f"Period {period} exceeds available periods for {year} (max: {max_period})")

        logger.info(f"Reading bands {band_offset + 1} to {band_offset + max_period} for {year} ({max_period} periods)")

        # Read all bands for this year at once (much more reliable than reading individually)
        # rasterio uses 1-based band indexing for read(window=..., indexes=...)
        band_list = list(range(band_offset + 1, band_offset + max_period + 1))
        vh_data = src_vh.read(indexes=band_list)  # (year_bands, height, width)
        profile = src_vh.profile.copy()
        vh_nodata = src_vh.nodata  # exclude (and avoid clipping) nodata in RG+SG mode
        if rg_sg:
            logger.info(f"RG+SG preprocessing ENABLED (clip+Savitzky-Golay); stack nodata={vh_nodata}")

        logger.info(f"VH data shape for {year}: {vh_data.shape}")

        height, width = src_vh.height, src_vh.width
        total_pixels = height * width

        # Load mask if provided
        mask = None
        if mask_file and os.path.exists(mask_file):
            logger.info(f"Loading mask from {mask_file}")
            with rasterio.open(mask_file) as mask_src:
                mask = mask_src.read(1)
                mask = mask > 0
                valid_pixels = np.sum(mask)
                logger.info(f"Mask: {valid_pixels:,}/{total_pixels:,} pixels valid ({100*valid_pixels/total_pixels:.1f}%)")
        else:
            mask = np.ones((height, width), dtype=bool)
            valid_pixels = total_pixels
            logger.info(f"No mask: processing all {total_pixels:,} pixels")

        # Test prediction if not skipped
        if not skip_test:
            logger.info("\n" + "-"*70)
            logger.info("Running test prediction on sample region...")
            logger.info("-"*70)

            test_mask = np.zeros((height, width), dtype=bool)
            center_r, center_c = height // 2, width // 2
            test_r_start = max(0, center_r - 50)
            test_r_end = min(height, center_r + 50)
            test_c_start = max(0, center_c - 50)
            test_c_end = min(width, center_c + 50)

            test_mask[test_r_start:test_r_end, test_c_start:test_c_end] = mask[test_r_start:test_r_end, test_c_start:test_c_end]

            test_start = time.time()

            test_features, test_valid_idx = extract_vh_features_vectorized(
                vh_data, period, test_mask,
                n_previous=FEATURE_PARAMS['N_PREVIOUS'],
                nodata=vh_nodata, rg_sg=rg_sg
            )

            if test_features is not None and len(test_features) > 0:
                test_scaled = scaler.transform(test_features)
                test_pred_prob = model.predict(test_scaled[:min(10, len(test_scaled))], verbose=0).flatten()
                test_pred = (test_pred_prob > 0.5).astype(int)

                test_time = time.time() - test_start

                logger.info(f"✓ Test prediction successful!")
                logger.info(f"  Test region: {test_features.shape[0]:,} valid pixels")
                logger.info(f"  Time: {test_time:.2f} seconds")
                logger.info(f"  Sample predictions: {test_pred}")
                logger.info(f"  Sample probabilities: {test_pred_prob}")
            else:
                logger.warning("⚠ No valid test features extracted")

        # Initialize output arrays
        predictions = np.full((height, width), -1, dtype=np.int8)
        confidences = np.full((height, width), -1.0, dtype=np.float32)

        # Process in chunks
        logger.info("\n" + "="*70)
        logger.info("VECTORIZED FEATURE EXTRACTION")
        logger.info("="*70)

        chunk_height = min(height, 500)
        n_chunks = (height + chunk_height - 1) // chunk_height

        logger.info(f"Processing image in {n_chunks} spatial chunks of ~{chunk_height} rows each")

        extraction_time = 0
        prediction_time = 0
        total_predicted = 0

        for chunk_idx in range(n_chunks):
            chunk_start_time = time.time()

            row_start = chunk_idx * chunk_height
            row_end = min((chunk_idx + 1) * chunk_height, height)

            logger.info(f"\n--- Chunk {chunk_idx+1}/{n_chunks}: Rows {row_start}-{row_end} ---")

            chunk_mask = mask[row_start:row_end, :]
            n_valid_in_chunk = np.sum(chunk_mask)

            if n_valid_in_chunk == 0:
                logger.info(f"  No valid pixels in chunk, skipping")
                continue

            logger.info(f"  Valid pixels in chunk: {n_valid_in_chunk:,}")

            chunk_vh = vh_data[:, row_start:row_end, :]

            extract_start = time.time()

            try:
                features, valid_indices = extract_vh_features_vectorized(
                    chunk_vh, period, chunk_mask,
                    n_previous=FEATURE_PARAMS['N_PREVIOUS'],
                    band_offset=0,  # Always 0 because vh_data is already subset for this year
                    nodata=vh_nodata, rg_sg=rg_sg
                )

                extract_time = time.time() - extract_start
                extraction_time += extract_time

                if features is None or len(features) == 0:
                    logger.warning(f"  No valid features extracted from chunk")
                    continue

                logger.info(f"  ✓ Extracted {features.shape[0]:,} feature vectors in {extract_time:.2f}s")

                features_scaled = scaler.transform(features)

                predict_start = time.time()

                # Single large-batch predict (calling model.predict per 10k-row
                # slice has huge per-call overhead; one call with a big batch is
                # ~10-50x faster, especially on GPU).
                all_confidences = model.predict(
                    features_scaled, batch_size=16384, verbose=0).flatten()
                all_predictions = (all_confidences > threshold).astype(np.int8)

                predict_time = time.time() - predict_start
                prediction_time += predict_time

                logger.info(f"  ✓ Predicted {len(all_predictions):,} pixels in {predict_time:.2f}s")

                # Assign to output arrays - use element-wise loop for safety
                rows_abs = row_start + valid_indices[0]
                cols = valid_indices[1]
                
                # Verify indices are in bounds
                assert np.all((rows_abs >= 0) & (rows_abs < height)), f"Row indices out of bounds"
                assert np.all((cols >= 0) & (cols < width)), f"Col indices out of bounds"
                
                # Debug: Log coordinate ranges for first chunk
                if chunk_idx == 0:
                    logger.info(f"  [DEBUG] Coordinate mapping: chunk rows [{row_start}, {row_end}), "
                               f"relative indices [{np.min(valid_indices[0])}, {np.max(valid_indices[0])}], "
                               f"absolute rows [{np.min(rows_abs)}, {np.max(rows_abs)}], "
                               f"cols [{np.min(cols)}, {np.max(cols)}]")
                    logger.info(f"  [DEBUG] First 10 coordinates (row, col): {list(zip(rows_abs[:10], cols[:10]))}")
                    logger.info(f"  [DEBUG] Last 10 coordinates (row, col): {list(zip(rows_abs[-10:], cols[-10:]))}")
                    logger.info(f"  [DEBUG] First 10 predictions: {all_predictions[:10]}")
                    logger.info(f"  [DEBUG] Last 10 predictions: {all_predictions[-10:]}")
                
                # Vectorized scatter (pixel coords are unique within a chunk).
                predictions[rows_abs, cols] = all_predictions
                confidences[rows_abs, cols] = all_confidences

                total_predicted += len(all_predictions)

                chunk_time = time.time() - chunk_start_time
                logger.info(f"  Chunk {chunk_idx+1} complete in {chunk_time:.2f}s")

            except Exception as e:
                logger.error(f"  Error processing chunk {chunk_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                continue

            tf.keras.backend.clear_session()
            gc.collect()

    total_time = time.time() - start_time

    valid_pred = predictions[predictions >= 0]

    logger.info("\n" + "="*70)
    logger.info("PREDICTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"  Feature extraction: {extraction_time:.2f}s ({100*extraction_time/total_time:.1f}%)")
    logger.info(f"  Model prediction: {prediction_time:.2f}s ({100*prediction_time/total_time:.1f}%)")
    logger.info(f"Total pixels predicted: {total_predicted:,}")

    if len(valid_pred) > 0:
        logger.info(f"\n{'='*70}")
        logger.info("CLASS DISTRIBUTION")
        logger.info("="*70)
        for class_id in [0, 1]:
            count = np.sum(predictions == class_id)
            pct = 100 * count / len(valid_pred)
            logger.info(f"  {class_id}. {CLASS_NAMES[class_id]}: {count:,} pixels ({pct:.1f}%)")

    return predictions, confidences, profile


def visualize_paddy(predictions, output_file):
    """Create visualization of paddy field map"""
    logger.info(f"\nCreating visualization...")

    colors = [(1, 1, 1), (0, 0.78, 0)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 10))

    plot_data = predictions.astype(float)
    plot_data[predictions == -1] = np.nan

    im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels([CLASS_NAMES[0], CLASS_NAMES[1]])

    ax.set_title('Binary Paddy Classification Map (VH-Only)', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Visualization saved to {output_file}")


def save_predictions(predictions, confidences, profile, output_dir, year, period):
    """Save prediction outputs"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nSaving outputs...")

    profile.update(count=1, dtype=np.int8, nodata=-1, compress='lzw')

    # Save predictions
    pred_file = output_dir / 'paddy_predictions.tif'
    with rasterio.open(pred_file, 'w', **profile) as dst:
        dst.write(predictions, 1)
    logger.info(f"✓ Predictions saved to {pred_file}")

    # Save confidences
    profile.update(dtype=np.float32, nodata=-1.0)
    conf_file = output_dir / 'confidence.tif'
    with rasterio.open(conf_file, 'w', **profile) as dst:
        dst.write(confidences, 1)
    logger.info(f"✓ Confidence saved to {conf_file}")

    # Save statistics
    valid_pred = predictions[predictions >= 0]
    valid_conf = confidences[confidences >= 0]

    stats_file = output_dir / 'statistics.txt'
    with open(stats_file, 'w') as f:
        f.write(f"VH-Only Binary Paddy Classification Statistics\n")
        f.write(f"Year: {year}, Period: {period}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total valid pixels: {len(valid_pred):,}\n\n")

        f.write("Class Distribution:\n")
        f.write("-" * 60 + "\n")
        for class_id in [0, 1]:
            count = np.sum(predictions == class_id)
            pct = 100 * count / len(valid_pred) if len(valid_pred) > 0 else 0
            f.write(f"{class_id}. {CLASS_NAMES[class_id]}: {count:,} ({pct:.2f}%)\n")

        paddy_pixels = int(np.sum(predictions == 1))
        # Pixel area from the actual geotransform. The stack is EPSG:4326
        # (degrees), so convert using the raster's center latitude
        # (~49.6 m x 50 m = 0.2481 ha at Java's ~7S). Earlier code wrongly
        # assumed 10 m pixels (0.01 ha), underestimating area ~25x.
        import math
        tr = profile['transform']
        crs = profile.get('crs')
        if crs is not None and crs.is_geographic:
            lat_c = tr.f + tr.e * (predictions.shape[0] / 2.0)
            m_per_deg_lat = 111320.0
            m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_c))
            px_area_ha = abs(tr.a) * m_per_deg_lon * abs(tr.e) * m_per_deg_lat / 10000.0
        else:
            px_area_ha = abs(tr.a) * abs(tr.e) / 10000.0  # projected CRS: already meters
        paddy_area_ha = paddy_pixels * px_area_ha
        f.write(f"\nPaddy Area Estimate:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Paddy pixels: {paddy_pixels:,}\n")
        f.write(f"Pixel size: {px_area_ha:.4f} ha/pixel\n")
        f.write(f"Estimated area: {paddy_area_ha:,.2f} hectares\n")

        if len(valid_conf) > 0:
            f.write(f"\nConfidence Statistics:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Mean: {np.mean(valid_conf):.3f}\n")
            f.write(f"Median: {np.median(valid_conf):.3f}\n")
            f.write(f"Std: {np.std(valid_conf):.3f}\n")
            f.write(f"Min: {np.min(valid_conf):.3f}\n")
            f.write(f"Max: {np.max(valid_conf):.3f}\n")

    logger.info(f"✓ Statistics saved to {stats_file}")

    # Visualize
    viz_file = output_dir / 'paddy_map.png'
    visualize_paddy(predictions, viz_file)


def main():
    parser = argparse.ArgumentParser(
        description='VH-only binary paddy prediction for multi-year stack'
    )
    parser.add_argument('--period', type=int, required=True,
                       help='Period number within the year (1-31)')
    parser.add_argument('--year', type=int, required=True, choices=[2023, 2024, 2025, 2026],
                       help='Year (2023-2026)')
    parser.add_argument('--model-dir', type=str, default=None,
                       help="Model artifacts dir (default: config model_files_paddy_vh). "
                            "Use model_files_paddy_vh_rg_sg for the RG+SG model.")
    parser.add_argument('--rg-sg', action='store_true',
                       help="Apply RG+SG preprocessing (clip + Savitzky-Golay) to match the "
                            "RG+SG model. REQUIRED when --model-dir is the RG+SG model.")
    parser.add_argument('--year-start-band', type=int, required=True,
                       help='Starting band for this year (1-based). E.g., 1 for 2023, 32 for 2024 if 2024 starts at band 32')
    parser.add_argument('--year-periods', type=int, required=True,
                       help='Number of periods in this year. E.g., 31 for 2023, 23 for 2024')
    parser.add_argument('--vh-stack', type=str, default=None,
                       help='Path to multi-year VH stack (default: from config)')
    parser.add_argument('--mask', type=str, default=None,
                       help='Mask file (optional)')
    parser.add_argument('--skip-test', action='store_true',
                       help='Skip test prediction')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: predictions_paddy/{year}/period_{period:02d})')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5). Higher values reduce false positives.')

    args = parser.parse_args()

    # Determine VH stack path
    vh_stack = args.vh_stack if args.vh_stack else FILES['PREDICTION_GEOTIFF_VH']

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"predictions_paddy/{args.year}/period_{args.period:02d}"

    try:
        logger.info("="*70)
        logger.info(f"MULTIYEAR VH-ONLY PADDY PREDICTION")
        logger.info(f"Year: {args.year}, Period: {args.period}")
        logger.info("="*70)

        # Load model
        model, scaler, feature_names = load_model_artifacts(model_dir=args.model_dir)

        # Create predictions
        logger.info(f"Classification threshold: {args.threshold}")
        predictions, confidences, profile = create_prediction_map_multiyear(
            vh_tif=vh_stack,
            model=model,
            scaler=scaler,
            period=args.period,
            year=args.year,
            year_start_band=args.year_start_band,
            year_periods=args.year_periods,
            mask_file=args.mask,
            skip_test=args.skip_test,
            threshold=args.threshold,
            rg_sg=args.rg_sg
        )

        # Save outputs
        save_predictions(predictions, confidences, profile, output_dir, args.year, args.period)

        logger.info("\n" + "="*70)
        logger.info("✓ PREDICTION COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nOutputs saved to: {output_dir}")

    except Exception as e:
        logger.error(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
