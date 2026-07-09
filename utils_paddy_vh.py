#!/usr/bin/env python3
"""
VH-Only Utilities for Binary Paddy Classification
Optimized vectorized implementation for fast feature extraction

Adapted from utils_dualpol_optimized.py to work with VH polarization only
for binary paddy vs non-paddy classification.

Key differences:
- Only VH features (no VV, no cross-pol)
- 29 features total (instead of 72)
- Binary classification output
- Optimized phenology thresholds for paddy detection
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Optional

try:
    from scipy.signal import savgol_filter
except ImportError:  # scipy optional; only needed for RG+SG preprocessing
    savgol_filter = None


# Default RG+SG preprocessing parameters. MUST stay in sync with
# extract_features_rg_sg.py so training and prediction preprocess identically.
RG_SG_CLIP = (-3500.0, -100.0)   # VH (dB x100) physical clip range
RG_SG_WINDOW = 5                 # Savitzky-Golay window_length
RG_SG_POLYORDER = 2              # Savitzky-Golay polyorder


def preprocess_vh_series(vh_values: np.ndarray,
                         clip_low: float = RG_SG_CLIP[0],
                         clip_high: float = RG_SG_CLIP[1],
                         sg_window: int = RG_SG_WINDOW,
                         sg_polyorder: int = RG_SG_POLYORDER,
                         smooth: bool = True,
                         clip: bool = True) -> np.ndarray:
    """Clip VH (dB x100) to a physical range, optionally Savitzky-Golay smooth,
    then re-clip. Shared by training feature extraction and prediction to
    guarantee no train/serve skew for the RG+SG model.

    Args:
        vh_values: (n_pixels, n_periods) raw VH time series (nodata already removed).
        clip_low/clip_high: physical bounds (VH backscatter is always negative;
            keeping it <= clip_high bounds ratio denominators away from zero).
        sg_window/sg_polyorder: Savitzky-Golay parameters.
        smooth: apply SG smoothing.
        clip: apply clipping.

    Returns:
        (n_pixels, n_periods) float32 preprocessed series.
    """
    vh = vh_values.astype(np.float64)
    if clip:
        vh = np.clip(vh, clip_low, clip_high)
    if smooth:
        if savgol_filter is None:
            raise ImportError("scipy is required for SG smoothing (pip install scipy)")
        if vh.shape[1] >= sg_window:
            vh = savgol_filter(vh, window_length=sg_window, polyorder=sg_polyorder, axis=1)
            if clip:
                # SG overshoots at window edges; re-clip so ratios stay bounded.
                vh = np.clip(vh, clip_low, clip_high)
    return vh.astype(np.float32)

# For Keras 3 compatibility
try:
    import keras
except ImportError:
    from tensorflow import keras


def extract_vh_features_vectorized(vh_data: np.ndarray,
                                   period: int,
                                   valid_mask: np.ndarray,
                                   n_previous: int = 6,
                                   band_offset: int = 0,
                                   nodata: Optional[float] = None,
                                   rg_sg: bool = False) -> Tuple[Optional[np.ndarray], Optional[Tuple]]:
    """
    Extract 29 VH-only features for ALL valid pixels at once using vectorized operations.

    This is optimized for binary paddy classification using only VH polarization.

    Args:
        vh_data: VH SAR data array (n_bands, height, width)
        period: Current period number (1-based, 7-31 for 2024)
        valid_mask: Boolean mask (height, width) indicating valid pixels
        n_previous: Number of previous periods to include (default: 6)
        band_offset: Offset to add to period for multi-year stacks (default: 0)
                    For 2024 data in 2023+2024 stack, use offset=31

    Returns:
        features: Array of shape (n_valid_pixels, 29) containing all VH features
                 29 = 7 temporal + 6 diff + 6 ratio + 10 phenology
        valid_indices: Tuple of (rows, cols) for valid pixels

    Expected speedup: 10-50x faster than pixel-by-pixel extraction
    """
    n_periods = n_previous + 1  # Total periods needed (7)
    height, width = vh_data.shape[1:]

    # Get band indices for backward time series
    # Apply band_offset to access correct year in multi-year stack
    band_indices = [period + band_offset - 1 - i for i in range(n_periods)]

    # Validate bands
    if min(band_indices) < 0 or max(band_indices) >= vh_data.shape[0]:
        raise ValueError(f"Invalid band indices for period {period} with offset {band_offset}: need bands {min(band_indices)} to {max(band_indices)}, have 0 to {vh_data.shape[0]-1}")

    # Extract all time periods at once: shape (n_periods, height, width)
    vh_stack = vh_data[band_indices]

    # Find valid pixels (no NaN/Inf in any time period)
    vh_valid = ~np.any(np.isnan(vh_stack) | np.isinf(vh_stack), axis=0)
    # Exclude nodata pixels too (e.g. RG stack -32768) so they are not clipped
    # into the valid range during RG+SG preprocessing.
    if nodata is not None:
        vh_valid &= ~np.any(vh_stack == nodata, axis=0)
    combined_valid = valid_mask & vh_valid

    valid_indices = np.where(combined_valid)
    n_valid = len(valid_indices[0])

    if n_valid == 0:
        return None, None

    print(f"Processing {n_valid:,} valid pixels using vectorized VH-only extraction...")
    
    # DEBUG: Check if indices are in expected order
    if n_valid > 0:
        print(f"[DEBUG] Valid indices - rows: [{valid_indices[0].min()}, {valid_indices[0].max()}], cols: [{valid_indices[1].min()}, {valid_indices[1].max()}]")
        print(f"[DEBUG] First 5 col indices: {valid_indices[1][:5]}")
        print(f"[DEBUG] Last 5 col indices: {valid_indices[1][-5:]}")

    # Extract values for valid pixels: shape (n_valid, n_periods=7)
    vh_values = vh_stack[:, valid_indices[0], valid_indices[1]].T  # (n_valid, 7)

    # RG+SG models are trained on clipped + Savitzky-Golay-smoothed VH. Apply the
    # IDENTICAL preprocessing here so prediction matches training (no train/serve
    # skew). Note: vh_values column 0 is t0 (most recent); SG is symmetric so the
    # ordering does not matter.
    if rg_sg:
        vh_values = preprocess_vh_series(vh_values)

    # ===== EXTRACT VH FEATURES (29) =====
    vh_features = extract_single_pol_features_vectorized(vh_values, 'VH', for_paddy=True)

    if vh_features is None:
        return None, None

    # Final validation: remove any pixels with NaN/Inf in features
    valid_features = ~np.any(np.isnan(vh_features) | np.isinf(vh_features), axis=1)

    if not np.all(valid_features):
        n_invalid = np.sum(~valid_features)
        print(f"Warning: {n_invalid} pixels had invalid features after extraction, removing them")
        vh_features = vh_features[valid_features]
        valid_indices = (valid_indices[0][valid_features], valid_indices[1][valid_features])

    print(f"Successfully extracted {vh_features.shape[0]:,} feature vectors ({vh_features.shape[1]} features each)")

    return vh_features, valid_indices


def extract_single_pol_features_vectorized(pol_values: np.ndarray,
                                          pol_name: str = 'VH',
                                          for_paddy: bool = True) -> Optional[np.ndarray]:
    """
    Extract 29 features from single polarization for ALL pixels at once.

    Features optimized for paddy detection:
    - 7 temporal values (t0 to t6)
    - 6 temporal differences
    - 6 temporal ratios
    - 10 phenology indicators (6 stages + 4 extrema)

    Args:
        pol_values: Array (n_pixels, 7) - temporal values for all pixels
        pol_name: Polarization name ('VH')
        for_paddy: If True, use paddy-optimized phenology thresholds

    Returns:
        features: Array (n_pixels, 29) or None if extraction fails
    """
    try:
        n_pixels = pol_values.shape[0]

        # ===== 1. TEMPORAL VALUES (7 features) =====
        # Already have pol_values: (n_pixels, 7)

        # ===== 2. TEMPORAL DIFFERENCES (6 features) =====
        # diff[i] = pol[i] - pol[i+1] (current minus previous)
        pol_diff = pol_values[:, :-1] - pol_values[:, 1:]  # (n_pixels, 6)

        # ===== 3. TEMPORAL RATIOS (6 features) =====
        # ratio[i] = pol[i] / pol[i+1] with safe division
        denominator = np.abs(pol_values[:, 1:])
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        pol_ratio = pol_values[:, :-1] / denominator  # (n_pixels, 6)

        # Handle any remaining NaN/Inf from ratios
        pol_ratio = np.nan_to_num(pol_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # ===== 4. PHENOLOGY INDICATORS (10 features: 6 indicators + 4 extrema) =====
        phenology_features = detect_paddy_phenology_vectorized(pol_values, pol_name)  # (n_pixels, 10)

        # ===== CONCATENATE ALL FEATURES =====
        # Total: 7 + 6 + 6 + 10 = 29 features
        features = np.concatenate([
            pol_values,         # 7 features: temporal values
            pol_diff,           # 6 features: differences
            pol_ratio,          # 6 features: ratios
            phenology_features  # 10 features: 6 indicators + 4 extrema
        ], axis=1).astype(np.float32)  # (n_pixels, 29)

        # Validate
        if features.shape[1] != 29:
            print(f"Warning: Expected 29 {pol_name} features, got {features.shape[1]}")
            return None

        return features

    except Exception as e:
        print(f"Error extracting {pol_name} features: {str(e)}")
        return None


def detect_paddy_phenology_vectorized(pol_values: np.ndarray,
                                     pol_name: str = 'VH') -> np.ndarray:
    """
    Detect paddy field phenology stages for ALL pixels at once using vectorized operations.

    Paddy-specific phenology thresholds (VH polarization, backscatter ×100):
    - Flooding/transplanting: Very low backscatter (specular reflection from water)
    - Early vegetative: Low-moderate backscatter (small plants emerging)
    - Late vegetative: Moderate backscatter (canopy development)
    - Reproductive: Higher backscatter (flowering, panicle formation)
    - Ripening: Moderate-high backscatter
    - Post-harvest: Return to low backscatter

    Args:
        pol_values: Backscatter time series (n_pixels, 7) scaled ×100
        pol_name: Polarization name ('VH')

    Returns:
        phenology_features: Array (n_pixels, 10)
            - 6 phenology stage indicators (binary)
            - 4 extrema features (continuous)
    """
    n_pixels = pol_values.shape[0]

    # Paddy-optimized thresholds for VH polarization (backscatter ×100)
    # These are more aggressive than general land cover thresholds
    if pol_name == 'VH':
        flooding_thresh = -2200      # Very low backscatter for flooding/transplanting
        early_veg_low = -2200
        early_veg_high = -1800       # Low backscatter for early growth
        late_veg_low = -1800
        late_veg_high = -1500        # Moderate backscatter for vegetative stage
        reproductive_low = -1500
        reproductive_high = -1200    # Higher backscatter for reproductive stage
        ripening_low = -1400
        ripening_high = -1100        # Moderate-high for ripening
        harvest_thresh = -1700       # Return to low after harvest
    else:  # Should not happen for paddy VH-only, but included for completeness
        flooding_thresh = -2000
        early_veg_low = -2000
        early_veg_high = -1600
        late_veg_low = -1600
        late_veg_high = -1300
        reproductive_low = -1300
        reproductive_high = -1000
        ripening_low = -1200
        ripening_high = -900
        harvest_thresh = -1500

    # Use current value (t0) for primary stage detection
    current_value = pol_values[:, 0]  # (n_pixels,)

    # Calculate temporal dynamics for better detection
    max_values = np.max(pol_values, axis=1)  # (n_pixels,)
    min_values = np.min(pol_values, axis=1)  # (n_pixels,)
    mean_values = np.mean(pol_values, axis=1)  # (n_pixels,)
    range_values = max_values - min_values  # (n_pixels,)

    # ===== DETECT PADDY-SPECIFIC PHENOLOGY STAGES =====

    # 1. Flooding/transplanting: Very low current backscatter
    flooding_detected = (current_value < flooding_thresh).astype(np.float32)

    # 2. Early vegetative: Low backscatter with increasing trend
    early_vegetative = ((current_value >= early_veg_low) &
                       (current_value < early_veg_high) &
                       (range_values > 200)).astype(np.float32)  # Some variation expected

    # 3. Late vegetative: Moderate backscatter with strong variation
    late_vegetative = ((current_value >= late_veg_low) &
                      (current_value < late_veg_high) &
                      (range_values > 300)).astype(np.float32)

    # 4. Reproductive stage: Higher backscatter, approaching peak
    reproductive = ((current_value >= reproductive_low) &
                   (current_value < reproductive_high) &
                   (current_value >= mean_values - 100)).astype(np.float32)  # Near mean or above

    # 5. Ripening: Moderate-high backscatter, may start declining
    ripening = ((current_value >= ripening_low) &
               (current_value < ripening_high)).astype(np.float32)

    # 6. Post-harvest: Low current but high historical max (evidence of previous crop)
    post_harvest = ((current_value < harvest_thresh) &
                   (max_values > late_veg_high) &
                   (range_values > 400)).astype(np.float32)  # Large variation = crop cycle

    # ===== EXTREMA FEATURES =====
    # These help identify the paddy growth cycle pattern
    min_indices = np.argmin(pol_values, axis=1) / 6.0  # Normalized to [0, 1]
    max_indices = np.argmax(pol_values, axis=1) / 6.0  # Normalized to [0, 1]

    # ===== STACK ALL FEATURES =====
    # Shape: (n_pixels, 10)
    phenology_features = np.stack([
        flooding_detected,   # Feature 0: Flooding/transplanting
        early_vegetative,    # Feature 1: Early vegetative growth
        late_vegetative,     # Feature 2: Late vegetative growth
        reproductive,        # Feature 3: Reproductive stage
        ripening,            # Feature 4: Ripening stage
        post_harvest,        # Feature 5: Post-harvest
        min_values,          # Feature 6: Minimum backscatter
        max_values,          # Feature 7: Maximum backscatter
        min_indices,         # Feature 8: Position of minimum (timing)
        max_indices          # Feature 9: Position of maximum (timing)
    ], axis=1).astype(np.float32)

    return phenology_features


def get_feature_names(n_previous: int = 6) -> list:
    """
    Get feature column names for VH-only features.

    Args:
        n_previous: Number of previous periods

    Returns:
        List of 29 feature names
    """
    n_periods = n_previous + 1
    features = []

    # VH features (29 total)
    pol = 'VH'

    # Temporal values (7)
    for i in range(n_periods):
        features.append(f'{pol}_t{i}')

    # Temporal differences (6)
    for i in range(n_periods - 1):
        features.append(f'{pol}_diff_t{i}_t{i+1}')

    # Temporal ratios (6)
    for i in range(n_periods - 1):
        features.append(f'{pol}_ratio_t{i}_t{i+1}')

    # Phenology indicators (6)
    features.extend([
        f'{pol}_flooding',
        f'{pol}_early_veg',
        f'{pol}_late_veg',
        f'{pol}_reproductive',
        f'{pol}_ripening',
        f'{pol}_post_harvest'
    ])

    # Extrema (4)
    features.extend([
        f'{pol}_min_value',
        f'{pol}_max_value',
        f'{pol}_min_idx_norm',
        f'{pol}_max_idx_norm'
    ])

    return features


def setup_gpu_strategy(min_memory_gb: int = 4):
    """
    Setup GPU strategy with memory configuration.

    Args:
        min_memory_gb: Minimum GPU memory required in GB

    Returns:
        TensorFlow distribution strategy
    """
    # Get list of GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("No GPU found. Using CPU.")
        return tf.distribute.OneDeviceStrategy("/cpu:0")

    print(f"Found {len(gpus)} GPU(s)")

    # Configure GPU memory growth
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {gpu}")
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Use MirroredStrategy for multi-GPU or OneDeviceStrategy for single GPU
    if len(gpus) > 1:
        print("Using MirroredStrategy for multiple GPUs")
        strategy = tf.distribute.MirroredStrategy()
    else:
        print(f"Using single GPU: {gpus[0]}")
        strategy = tf.distribute.OneDeviceStrategy(f"/gpu:0")

    return strategy


# Custom Keras layer for temperature scaling
class TemperatureLayer(keras.layers.Layer):
    """Temperature scaling layer for calibrated predictions"""

    def __init__(self, temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        return inputs / self.temperature

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Register the custom layer
keras.saving.register_keras_serializable(package="Custom", name="TemperatureLayer")(TemperatureLayer)


def create_paddy_vh_model(input_dim: int = 29, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Create MLP model for VH-only binary paddy classification.

    Architecture optimized for 29-feature input:
    - Input: 29 features (VH-only)
    - Hidden layers with batch normalization and dropout
    - Temperature-scaled sigmoid output
    - Output: Binary (paddy vs non-paddy)

    Args:
        input_dim: Number of input features (default: 29)
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        # First hidden layer
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.Dropout(0.3),

        # Second hidden layer
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.Dropout(0.3),

        # Third hidden layer
        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(negative_slope=0.1),
        tf.keras.layers.Dropout(0.2),

        # Output layer with temperature scaling for binary classification
        tf.keras.layers.Dense(1),
        TemperatureLayer(temperature=0.5),
        tf.keras.layers.Activation('sigmoid')
    ])

    # Compile model with SGD optimizer (avoids XLA JIT issues with Adam)
    # SGD with momentum works well for this task and doesn't trigger XLA Pow errors
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')],
        jit_compile=False  # Disable XLA JIT compilation
    )

    return model


if __name__ == "__main__":
    print("VH-Only Binary Paddy Classification Feature Extraction Utilities")
    print(f"Total features: 29 (VH-only)")
    print("\nVectorized operations for 10-50x speedup!")
    print("\nFeature names:")
    feature_names = get_feature_names()
    for i, name in enumerate(feature_names, 1):
        print(f"{i:3d}. {name}")
