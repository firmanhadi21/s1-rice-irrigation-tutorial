#!/usr/bin/env python3
"""
VH-Only Binary Paddy Classification - Training Script with SMOTE

Uses SMOTE to balance the severely imbalanced training data:
- Original: 92% paddy, 8% non-paddy
- After SMOTE: 50% paddy, 50% non-paddy

This should significantly improve the model's ability to distinguish paddy from non-paddy.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# TensorFlow environment setup - Force CPU only to avoid GPU/XLA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false --tf_xla_auto_jit=0'
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
os.environ['TF_DETERMINISTIC_OPS'] = "1"

import tensorflow as tf
tf.random.set_seed(42)
tf.config.optimizer.set_jit(False)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import logging
import gc

# Import configurations
from config_paddy_vh import (
    PATHS, FILES, MODEL_FILES, FEATURE_PARAMS, TRAINING_PARAMS, NUM_CLASSES,
    CLASS_NAMES, setup_directories
)

# Import VH-only utilities
from utils_paddy_vh import (
    get_feature_names,
    create_paddy_vh_model,
    TemperatureLayer
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train VH-only binary paddy classifier with SMOTE balancing.")
    ap.add_argument(
        '--training-data', default='model_files_paddy_vh/training_data_opensea.csv',
        help="Training CSV (35-col schema: metadata + 29 VH features + paddy).")
    ap.add_argument(
        '--model-dir', default=None,
        help="Output dir for model artifacts. Default: config PATHS['MODEL_DIR'].")
    return ap.parse_args()


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("VH-ONLY BINARY PADDY CLASSIFICATION WITH SMOTE")
    logger.info("="*80)

    # Output directory for model artifacts (versioned dir keeps the original
    # LC-trained model intact when training the RG+SG variant).
    model_dir = args.model_dir if args.model_dir else PATHS['MODEL_DIR']
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, os.path.basename(MODEL_FILES['MODEL']))
    scaler_path = os.path.join(model_dir, os.path.basename(MODEL_FILES['SCALER']))

    # Load pre-extracted training data
    training_data_file = args.training_data
    logger.info(f"\nLoading training data from: {training_data_file}")
    logger.info(f"Model artifacts will be saved to: {model_dir}")

    df = pd.read_csv(training_data_file)
    logger.info(f"Loaded {len(df)} samples")

    # Show original class distribution
    logger.info("\nOriginal class distribution:")
    for class_id in [0, 1]:
        count = sum(df['paddy'] == class_id)
        pct = 100 * count / len(df)
        logger.info(f"  {CLASS_NAMES[class_id]}: {count} ({pct:.1f}%)")

    # Separate features and labels
    feature_names = get_feature_names(n_previous=FEATURE_PARAMS['N_PREVIOUS'])
    X = df[feature_names].values
    y = df['paddy'].values

    logger.info(f"\nOriginal feature matrix shape: {X.shape}")
    logger.info(f"Original labels shape: {y.shape}")

    # Check for NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Found NaN/Inf in features, removing...")
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.any(np.isinf(X), axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        logger.info(f"After cleaning: {len(X)} samples")

    # Apply SMOTE to balance classes
    logger.info("\n" + "="*80)
    logger.info("APPLYING SMOTE")
    logger.info("="*80)
    logger.info("This will create synthetic non-paddy samples to balance the dataset...")

    # Use SMOTE with k_neighbors=5 (default)
    # If non-paddy samples < 6, SMOTE will fail, so check first
    n_non_paddy = sum(y == 0)
    if n_non_paddy < 6:
        logger.error(f"Not enough non-paddy samples ({n_non_paddy}) for SMOTE!")
        logger.error("Need at least 6 non-paddy samples. Please add more training data.")
        sys.exit(1)

    # Adjust k_neighbors if needed
    k_neighbors = min(5, n_non_paddy - 1)
    logger.info(f"Using k_neighbors={k_neighbors} for SMOTE")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info(f"\nAfter SMOTE:")
    logger.info(f"  Feature matrix shape: {X_resampled.shape}")
    logger.info(f"  Total samples: {len(y_resampled)}")
    logger.info(f"  Class distribution:")
    for class_id in [0, 1]:
        count = sum(y_resampled == class_id)
        pct = 100 * count / len(y_resampled)
        logger.info(f"    {CLASS_NAMES[class_id]}: {count} ({pct:.1f}%)")

    # Scale features
    logger.info("\nScaling features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Cross-validation
    n_folds = TRAINING_PARAMS['N_FOLDS']
    logger.info(f"\n{'='*80}")
    logger.info(f"{n_folds}-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*80}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    fold_auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_resampled), 1):
        logger.info(f"\nFold {fold_idx}/{n_folds}")
        logger.info("-"*80)

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")

        # Create model
        model = create_paddy_vh_model(
            input_dim=FEATURE_PARAMS['TOTAL_FEATURES'],
            learning_rate=TRAINING_PARAMS['LEARNING_RATE']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=TRAINING_PARAMS['EARLY_STOPPING_PATIENCE'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-6
            )
        ]

        # Train (no class weights needed since SMOTE balanced the data)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TRAINING_PARAMS['EPOCHS'],
            batch_size=TRAINING_PARAMS['BATCH_SIZE'],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        y_pred_prob = model.predict(X_val, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_prob)

        fold_scores.append(acc)
        fold_auc_scores.append(auc)

        logger.info(f"\nFold {fold_idx} Results:")
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")

        logger.info(f"\nConfusion Matrix:")
        logger.info(confusion_matrix(y_val, y_pred))

        logger.info(f"\nClassification Report:")
        target_names = [CLASS_NAMES[i] for i in [0, 1]]
        logger.info("\n" + classification_report(y_val, y_pred, target_names=target_names))

        # Clear session
        tf.keras.backend.clear_session()
        gc.collect()

    # Print CV summary
    logger.info("\n" + "="*80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Mean Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    logger.info(f"Mean AUC: {np.mean(fold_auc_scores):.4f} ± {np.std(fold_auc_scores):.4f}")
    logger.info(f"Fold Accuracies: {[f'{s:.4f}' for s in fold_scores]}")
    logger.info(f"Fold AUC Scores: {[f'{s:.4f}' for s in fold_auc_scores]}")

    # Train final model on all SMOTE-balanced data
    logger.info("\n" + "="*80)
    logger.info("TRAINING FINAL MODEL ON ALL SMOTE-BALANCED DATA")
    logger.info("="*80)

    final_model = create_paddy_vh_model(
        input_dim=FEATURE_PARAMS['TOTAL_FEATURES'],
        learning_rate=TRAINING_PARAMS['LEARNING_RATE']
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=TRAINING_PARAMS['EARLY_STOPPING_PATIENCE'],
        restore_best_weights=True,
        verbose=1
    )

    history = final_model.fit(
        X_scaled, y_resampled,
        epochs=TRAINING_PARAMS['EPOCHS'],
        batch_size=TRAINING_PARAMS['BATCH_SIZE'],
        callbacks=[early_stop],
        verbose=1
    )

    # Save model and artifacts (model_path / scaler_path resolved at top of main)
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"\nSaving model to: {model_path}")
    final_model.save(model_path)

    logger.info(f"Saving scaler to: {scaler_path}")
    joblib.dump(scaler, scaler_path)

    # Save feature names
    feature_names_path = os.path.join(model_dir, 'feature_columns.txt')
    with open(feature_names_path, 'w') as f:
        for fname in feature_names:
            f.write(f"{fname}\n")
    logger.info(f"Saved feature names to: {feature_names_path}")

    # Save class names
    class_names_path = os.path.join(model_dir, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for class_id in [0, 1]:
            f.write(f"{class_id}\n")
    logger.info(f"Saved class names to: {class_names_path}")

    # Save training info
    info_path = os.path.join(model_dir, 'training_info_smote.txt')
    with open(info_path, 'w') as f:
        f.write("VH-Only Binary Paddy Classification - SMOTE Training\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {training_data_file}\n\n")
        f.write("Original Class Distribution:\n")
        f.write(f"  Paddy: {sum(y==1)} ({100*sum(y==1)/len(y):.1f}%)\n")
        f.write(f"  Non-Paddy: {sum(y==0)} ({100*sum(y==0)/len(y):.1f}%)\n\n")
        f.write("After SMOTE:\n")
        f.write(f"  Paddy: {sum(y_resampled==1)} ({100*sum(y_resampled==1)/len(y_resampled):.1f}%)\n")
        f.write(f"  Non-Paddy: {sum(y_resampled==0)} ({100*sum(y_resampled==0)/len(y_resampled):.1f}%)\n\n")
        f.write(f"Cross-Validation Results:\n")
        f.write(f"  Mean Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}\n")
        f.write(f"  Mean AUC: {np.mean(fold_auc_scores):.4f} ± {np.std(fold_auc_scores):.4f}\n")
    logger.info(f"Saved training info to: {info_path}")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nModel artifacts saved to: {model_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Test predictions on 2023 and 2024 data")
    logger.info("  2. Compare results with previous model")
    logger.info("  3. If results are good, generate full time series")


if __name__ == '__main__':
    main()
