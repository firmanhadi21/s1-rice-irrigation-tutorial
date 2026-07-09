#!/usr/bin/env python3
"""
Configuration for VH-Only Binary Paddy Classification

This configuration is adapted from the dual-polarization land cover system
to work with VH-only data for binary paddy detection.

Key differences from dual-pol:
- Only VH polarization (no VV)
- Binary classification (paddy vs non-paddy)
- 29 features instead of 72
"""

import os
from pathlib import Path

# ===== PATHS =====
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
STACKS_DIR = DATA_DIR / 'stacks'
MODEL_DIR = BASE_DIR / 'model_files_paddy_vh'
PREDICTIONS_DIR = BASE_DIR / 'predictions_paddy'

PATHS = {
    'BASE_DIR': str(BASE_DIR),
    'DATA_DIR': str(DATA_DIR),
    'STACKS_DIR': str(STACKS_DIR),
    'MODEL_DIR': str(MODEL_DIR),
    'PREDICTIONS_DIR': str(PREDICTIONS_DIR)
}

# ===== DATA FILES =====
# Training data
TRAINING_CSV = DATA_DIR / 'training_points_paddy_binary_combined.csv'  # Combined: original + new survey data
TRAINING_STACK_VH = STACKS_DIR / 'stack_vh_2023_2024.tif'  # Multi-year stack (54 bands: 31 from 2023 + 23 from 2024)
PERIOD_LOOKUP = DATA_DIR / 'period_lookup_2024.csv'

# Prediction data
PREDICTION_STACK_VH = STACKS_DIR / 'stack_vh_2023_2024.tif'  # Multi-year stack (54 bands)

FILES = {
    'TRAINING_CSV': str(TRAINING_CSV),
    'TRAINING_GEOTIFF_VH': str(TRAINING_STACK_VH),
    'PERIOD_LOOKUP': str(PERIOD_LOOKUP),
    'PREDICTION_GEOTIFF_VH': str(PREDICTION_STACK_VH)
}

# ===== MODEL FILES =====
MODEL_FILES = {
    'MODEL': str(MODEL_DIR / 'paddy_vh_model.keras'),
    'SCALER': str(MODEL_DIR / 'scaler.joblib'),
    'FEATURE_COLUMNS': str(MODEL_DIR / 'feature_columns.txt'),
    'CLASS_NAMES': str(MODEL_DIR / 'class_names.txt')
}

# ===== FEATURE PARAMETERS =====
FEATURE_PARAMS = {
    'N_PREVIOUS': 6,           # Number of previous periods (backward-looking)
    'N_PERIODS': 7,            # Total periods (current + 6 previous)
    'MIN_PERIOD': 7,           # Minimum period for training/prediction
    'MAX_PERIOD': 54,          # Maximum period (54 bands total in multi-year stack)
    'BAND_OFFSET': 31,         # Offset for 2024 periods in multi-year stack (31 bands from 2023)
    'VH_FEATURES': 29,         # 7 temporal + 6 diff + 6 ratio + 10 phenology
    'TOTAL_FEATURES': 29       # VH only (no VV, no cross-pol)
}

# ===== TRAINING PARAMETERS =====
TRAINING_PARAMS = {
    'EPOCHS': 200,
    'BATCH_SIZE': 128,
    'LEARNING_RATE': 0.001,
    'N_FOLDS': 5,                      # Cross-validation folds
    'EARLY_STOPPING_PATIENCE': 15,
    'USE_CLASS_WEIGHTS': True          # Balance paddy/non-paddy classes
}

# ===== PREDICTION PARAMETERS =====
PREDICTION_PARAMS = {
    'CHUNK_SIZE': 500,          # Process 500 rows at a time
    'BATCH_SIZE': 512,          # Prediction batch size
    'CONFIDENCE_THRESHOLD': 0.5 # Binary classification threshold
}

# ===== CLASS DEFINITIONS =====
# Binary classification: 0 = non-paddy, 1 = paddy
NUM_CLASSES = 2

CLASS_NAMES = {
    0: 'Non-Paddy',
    1: 'Paddy'
}

CLASS_COLORS = {
    0: (255, 255, 255),  # White for non-paddy
    1: (0, 200, 0)       # Green for paddy
}

# ===== UTILITY FUNCTIONS =====

def setup_directories():
    """Create necessary directories if they don't exist"""
    for path_key in ['DATA_DIR', 'STACKS_DIR', 'MODEL_DIR', 'PREDICTIONS_DIR']:
        path = Path(PATHS[path_key])
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {path}")


def verify_files(for_training=True):
    """
    Verify required files exist

    Args:
        for_training: If True, check training files; if False, check prediction files
    """
    required_files = []

    if for_training:
        required_files = [
            ('Training CSV', FILES['TRAINING_CSV']),
            ('VH Training Stack', FILES['TRAINING_GEOTIFF_VH']),
            ('Period Lookup', FILES['PERIOD_LOOKUP'])
        ]
    else:
        required_files = [
            ('VH Prediction Stack', FILES['PREDICTION_GEOTIFF_VH']),
            ('Period Lookup', FILES['PERIOD_LOOKUP']),
            ('Trained Model', MODEL_FILES['MODEL']),
            ('Scaler', MODEL_FILES['SCALER']),
            ('Feature Columns', MODEL_FILES['FEATURE_COLUMNS'])
        ]

    all_exist = True
    for name, filepath in required_files:
        if os.path.exists(filepath):
            print(f"✓ {name}: {filepath}")
        else:
            print(f"✗ {name} NOT FOUND: {filepath}")
            all_exist = False

    return all_exist


def get_period_directory(period):
    """Get output directory for a specific period"""
    period_dir = Path(PATHS['PREDICTIONS_DIR']) / f'period_{period:02d}'
    period_dir.mkdir(parents=True, exist_ok=True)
    return str(period_dir)


def get_prediction_files(period):
    """Get output file paths for a specific period"""
    period_dir = get_period_directory(period)

    return {
        'predictions': os.path.join(period_dir, 'paddy_predictions.tif'),
        'confidence': os.path.join(period_dir, 'confidence.tif'),
        'visualization': os.path.join(period_dir, 'paddy_map.png'),
        'statistics': os.path.join(period_dir, 'statistics.txt')
    }


def save_class_names():
    """Save class names to file"""
    with open(MODEL_FILES['CLASS_NAMES'], 'w') as f:
        for class_id in sorted(CLASS_NAMES.keys()):
            f.write(f"{class_id},{CLASS_NAMES[class_id]}\n")
    print(f"✓ Class names saved to {MODEL_FILES['CLASS_NAMES']}")


def load_class_names():
    """Load class names from file"""
    class_names = {}
    with open(MODEL_FILES['CLASS_NAMES'], 'r') as f:
        for line in f:
            class_id, name = line.strip().split(',')
            class_names[int(class_id)] = name
    return class_names


def print_config_info():
    """Print configuration summary"""
    print("\n" + "="*80)
    print("VH-ONLY BINARY PADDY CLASSIFICATION CONFIGURATION")
    print("="*80)

    print("\nClassification Task:")
    print(f"  Type: Binary (Paddy vs Non-Paddy)")
    print(f"  Polarization: VH only")
    print(f"  Classes: {NUM_CLASSES}")
    for class_id, name in CLASS_NAMES.items():
        print(f"    {class_id}: {name}")

    print("\nFeature Configuration:")
    print(f"  Temporal periods: {FEATURE_PARAMS['N_PERIODS']} (current + {FEATURE_PARAMS['N_PREVIOUS']} previous)")
    print(f"  VH features: {FEATURE_PARAMS['VH_FEATURES']}")
    print(f"    - 7 temporal values")
    print(f"    - 6 temporal differences")
    print(f"    - 6 temporal ratios")
    print(f"    - 10 phenology indicators")
    print(f"  Total features: {FEATURE_PARAMS['TOTAL_FEATURES']}")

    print("\nTraining Configuration:")
    print(f"  Epochs: {TRAINING_PARAMS['EPOCHS']}")
    print(f"  Batch size: {TRAINING_PARAMS['BATCH_SIZE']}")
    print(f"  Learning rate: {TRAINING_PARAMS['LEARNING_RATE']}")
    print(f"  Cross-validation folds: {TRAINING_PARAMS['N_FOLDS']}")
    print(f"  Early stopping patience: {TRAINING_PARAMS['EARLY_STOPPING_PATIENCE']}")
    print(f"  Class weights: {TRAINING_PARAMS['USE_CLASS_WEIGHTS']}")

    print("\nData Paths:")
    print(f"  Training CSV: {FILES['TRAINING_CSV']}")
    print(f"  VH Stack: {FILES['TRAINING_GEOTIFF_VH']}")
    print(f"  Model directory: {PATHS['MODEL_DIR']}")
    print(f"  Predictions directory: {PATHS['PREDICTIONS_DIR']}")

    print("="*80 + "\n")


if __name__ == '__main__':
    print_config_info()

    print("\nSetting up directories...")
    setup_directories()

    print("\nVerifying training files...")
    verify_files(for_training=True)

    print("\n" + "="*80)
    print("Configuration module loaded successfully!")
    print("="*80)
