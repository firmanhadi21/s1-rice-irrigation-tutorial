#!/usr/bin/env bash
# =============================================================================
# Smoke-test the hands-on tutorial workflow.
# Run on a machine that has the Python geo/ML env (and, for the SAR steps, a
# Sentinel-1 VH stack). Reports PASS / FAIL / SKIP per step and a final summary.
#
# Run from the REPO ROOT:
#   conda activate <your-geo-ml-env>
#   bash tutorial/scripts/test_tutorial.sh                       # core steps only
#   VH_STACK=data/sample/klambu_vh_2024.tif VH_PERIODS=18 \      # + SAR steps
#     bash tutorial/scripts/test_tutorial.sh
#
# Env overrides: PY (python), VH_STACK, VH_PERIODS, MASK, DI_GPKG, PRED_DIR.
# NOTE: the train step overwrites model_files_paddy_vh/*; `git checkout` after if needed.
# =============================================================================
set -u
PY=${PY:-python}
VH_STACK=${VH_STACK:-}
VH_PERIODS=${VH_PERIODS:-18}
MASK=${MASK:-}
DI_GPKG=${DI_GPKG:-2026/irrigation_performance/klambu.gpkg}
PRED_DIR=${PRED_DIR:-predictions_sample}
export CUDA_VISIBLE_DEVICES=-1

pass=0; fail=0; skip=0
hdr(){ printf "\n\033[1m=== %s ===\033[0m\n" "$1"; }
ok(){ echo "  [PASS] $1"; pass=$((pass+1)); }
no(){ echo "  [FAIL] $1"; fail=$((fail+1)); }
sk(){ echo "  [SKIP] $1"; skip=$((skip+1)); }
have(){ [ -s "$1" ]; }

cd "$(dirname "$0")/../.." || { echo "cannot cd to repo root"; exit 2; }
echo "repo: $(pwd)"
echo "python: $($PY -c 'import sys;print(sys.executable)' 2>/dev/null)"

# -----------------------------------------------------------------------------
hdr "0. dependencies"
$PY - <<'PYEOF'
import importlib.util as u, sys
need=["numpy","pandas","scipy","sklearn","imblearn","tensorflow",
      "rasterio","geopandas","matplotlib","joblib"]
opt=["osgeo","ee"]
miss=[m for m in need if u.find_spec(m) is None]
optm=[m for m in opt if u.find_spec(m) is None]
print("  required missing:", ",".join(miss) or "none")
print("  optional missing:", ",".join(optm) or "none")
sys.exit(1 if miss else 0)
PYEOF
[ $? -eq 0 ] && ok "required deps present" || no "missing required deps (install them first)"

# -----------------------------------------------------------------------------
hdr "1. train paddy model (train_paddy_vh_smote.py) — from shipped CSV"
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  sk "training skipped (SKIP_TRAIN=1)"
elif ls model_files_paddy_vh/training_data_*.csv >/dev/null 2>&1; then
  if $PY train_paddy_vh_smote.py > /tmp/tt_train.log 2>&1; then
    have model_files_paddy_vh/paddy_vh_model.keras && ok "model written" || no "no model file"
    grep -iE "accuracy|auc" /tmp/tt_train.log | tail -3
  else
    no "train_paddy_vh_smote.py exited non-zero (see /tmp/tt_train.log)"; tail -5 /tmp/tt_train.log
  fi
else
  sk "no training_data_*.csv found"
fi

# -----------------------------------------------------------------------------
hdr "2. irrigation maps (make_irrigation_maps.py) — from shipped block CSVs"
if have "$DI_GPKG" && have 2026/irrigation_performance/uniformity_block_results.csv; then
  mkdir -p results_csv
  cp 2026/irrigation_performance/uniformity_block_results.csv results_csv/ 2>/dev/null
  cp 2026/irrigation_performance/reliability_block_results.csv results_csv/ 2>/dev/null
  if $PY make_irrigation_maps.py > /tmp/tt_irrmap.log 2>&1; then
    have 2026/figures/fig_irrigation_klambu_maps.png && ok "SI/CU/RI map rendered" || no "no map png"
  else
    no "make_irrigation_maps.py failed (see /tmp/tt_irrmap.log)"; tail -5 /tmp/tt_irrmap.log
  fi
else
  sk "klambu.gpkg or block CSVs missing"
fi

# -----------------------------------------------------------------------------
hdr "3. predict paddy (predict_paddy_vh_multiyear.py) — needs a VH stack"
if [ -n "$VH_STACK" ] && have "$VH_STACK"; then
  MASK_ARG=""; [ -n "$MASK" ] && have "$MASK" && MASK_ARG="--mask $MASK"
  if have "$PRED_DIR/2024/period_15/paddy_predictions.tif"; then
    ok "paddy_predictions.tif already present (reused)"
  elif $PY predict_paddy_vh_multiyear.py --period 15 --year 2024 \
        --year-start-band 1 --year-periods "$VH_PERIODS" \
        --vh-stack "$VH_STACK" $MASK_ARG --skip-test \
        --output-dir "$PRED_DIR/2024/period_15" > /tmp/tt_pred.log 2>&1; then
    have "$PRED_DIR/2024/period_15/paddy_predictions.tif" && ok "paddy_predictions.tif written" \
        || no "no prediction raster"
  else
    no "predict failed (see /tmp/tt_pred.log)"; tail -8 /tmp/tt_pred.log
  fi
else
  sk "set VH_STACK=... to test prediction (e.g. run fetch_sample_aoi.py first)"
fi

# -----------------------------------------------------------------------------
hdr "4. multi-year consensus (create_multiyear_composite_paddy.py)"
if have "$PRED_DIR/2024/period_15/paddy_predictions.tif"; then
  # need a couple more periods for a consensus; skip any already predicted (fast re-runs)
  for p in 13 17; do
    if have "$PRED_DIR/2024/period_$p/paddy_predictions.tif"; then
      echo "  (period $p already predicted, reusing)"
    else
      echo "  predicting period $p ..."
      $PY predict_paddy_vh_multiyear.py --period $p --year 2024 \
        --year-start-band 1 --year-periods "$VH_PERIODS" \
        --vh-stack "$VH_STACK" ${MASK:+--mask $MASK} --skip-test \
        --output-dir "$PRED_DIR/2024/period_$p" >/dev/null 2>&1
    fi
  done
  if $PY create_multiyear_composite_paddy.py --years 2024 \
        --periods 2024:"13 15 17" --predictions-dir "$PRED_DIR" \
        --output consensus_sample --min-years 1 --min-detections 2 \
        --min-confidence 0.7 --min-mean-confidence 0.6 > /tmp/tt_cons.log 2>&1; then
    have consensus_sample/consensus_paddy_map.tif && ok "consensus map written" \
        || no "no consensus raster"
  else
    no "consensus failed (see /tmp/tt_cons.log)"; tail -6 /tmp/tt_cons.log
  fi
else
  sk "no predictions to build a consensus from"
fi

# -----------------------------------------------------------------------------
hdr "5. irrigation indices (calculate_irrigation_indices.py)"
if have "$PRED_DIR/2024/period_15/paddy_predictions.tif" && have "$DI_GPKG"; then
  if $PY calculate_irrigation_indices.py --predictions-dir "$PRED_DIR/2024" \
        --periods 13 15 17 --di-boundary "$DI_GPKG" \
        --output-dir irrigation_results/klambu_test > /tmp/tt_idx.log 2>&1; then
    ls irrigation_results/klambu_test/*.csv >/dev/null 2>&1 && ok "index CSVs written" \
        || no "no index outputs"
  else
    no "calculate_irrigation_indices.py failed (see /tmp/tt_idx.log)"; tail -8 /tmp/tt_idx.log
  fi
else
  sk "need predictions + DI boundary"
fi

# -----------------------------------------------------------------------------
hdr "6. cropping intensity (cropping_intensity.py --mode validate)"
echo "  note: full mode + produce_kc_map.py hardcode the HPC stack path"
echo "        (STACK=.../java_vh_2024_2026_50m.tif). Edit those constants to test full mode."
if $PY cropping_intensity.py --mode validate > /tmp/tt_ci.log 2>&1; then
  ok "cycle-counter validation ran"; grep -iE "CI=|mean detected" /tmp/tt_ci.log | tail -3
else
  sk "validate needs the hardcoded STACK/CIMAP paths (HPC only); see /tmp/tt_ci.log"
fi

# -----------------------------------------------------------------------------
hdr "7. crop coefficient (produce_kc_map.py) — needs a VH stack + paddy mask"
if [ -n "$VH_STACK" ] && have "$VH_STACK" && have consensus_sample/consensus_paddy_map.tif; then
  if $PY produce_kc_map.py --period-band 14 --out kc_test \
        --stack "$VH_STACK" --paddy-mask consensus_sample/consensus_paddy_map.tif \
        > /tmp/tt_kc.log 2>&1; then
    ls kc_test/kc_band*.tif >/dev/null 2>&1 && ok "Kc map written" || no "no Kc raster"
    grep -iE "Kc .*mean=|active-crop" /tmp/tt_kc.log | tail -2
  else
    no "produce_kc_map.py failed (see /tmp/tt_kc.log)"; tail -6 /tmp/tt_kc.log
  fi
else
  sk "need VH_STACK + consensus_sample/consensus_paddy_map.tif (run steps 3-4 first)"
fi

# -----------------------------------------------------------------------------
hdr "SUMMARY"
echo "  PASS=$pass  FAIL=$fail  SKIP=$skip"
[ $fail -eq 0 ] && echo "  -> no failures" || echo "  -> $fail failure(s): paste the [FAIL] blocks + /tmp/tt_*.log tails"
exit $fail
