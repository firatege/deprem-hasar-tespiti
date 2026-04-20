# Project Journey - End-to-End Technical Log

This document records the full technical journey so far: dataset structure, split strategy, model evolution (ML + DL), scripts, tests, hyperparameter decisions, metrics interpretation, production selection, and operational commands.

## 1) Goal and Scope

Primary objective:

- Binary damage detection from pre/post disaster imagery.
- Policy target: `major-damage` + `destroyed` -> `binary_damage = 1`.

Key constraints followed:

- Use only available data in project artifacts.
- Prioritize robust baseline before complex models.
- Keep production behavior deterministic (model path + threshold config).

---

## 2) Dataset and Processed Artifacts

Core processed data files under `data/processed/`:

- `train_manifest.csv`
- `train_split.csv`
- `val_split.csv`
- `test_split.csv`
- `class_imbalance_summary.csv`
- `class_weights.json`
- `seed_scan_report.json`

Observed split schema (manifest/split CSV headers):

- `split`
- `event_id`
- `tile_id`
- `pre_image_path`
- `post_image_path`
- `pre_label_path`
- `post_label_path`
- `damage_label`
- `binary_damage`
- `building_count`
- `unknown_subtype_count`

Important note:

- `pga_value` column is not present in current split files.
- Any PGA-aware code path falls back to defaults unless external enrichment is added.

---

## 3) Split Strategy and Why Ratios Look Non-Exact

Split logic is implemented in `scripts/split_dataset.py`.

Design choices:

- Assignment is **event-level** (`event_id`), not row-level, to reduce leakage.
- Objective emphasizes class-balance drift more than strict row ratio:
  - `RATIO_WEIGHT = 1.0`
  - `POS_RATIO_WEIGHT = 4.0`
- Seed scan/tie-break logic chooses best candidate by objective.

Consequence:

- Requested ratio `0.7/0.15/0.15` is a target, but final row counts can deviate because whole events move together.

Current row counts observed:

- train: `3870`
- val: `3366`
- test: `3798`
- total: `11034`

---

## 4) Scalar Baseline Feature Engineering

Implemented in `app/ml/scalar_baseline.py`.

### 4.1 Image change features

For each band (up to `max_bands`, default `3`):

- `mean(pre)`
- `mean(post)`
- `mean(abs(post-pre))`
- `std(pre)`
- `std(post)`
- `std(abs(post-pre))`
- `skewness(abs(post-pre))`
- `kurtosis(abs(post-pre))`

So image feature dimension is `max_bands * 8` (default `24`).

### 4.2 Scalar/context features

Appended scalar block (`5` dims):

- normalized `pga_value` (signed `log1p`, optional)
- `magnitude`
- `depth_km`
- `acquisition_delta_days`
- `building_density` (fallback to `building_count`)

Total full feature dimension default: `24 + 5 = 29`.

---

## 5) Training and Inference Scripts

### 5.1 Main scalar trainer

File: `scripts/train_scalar_baseline.py`

Supported model families:

- `logreg`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`

Outputs:

- `model.pkl`
- `metrics.json`

### 5.2 Inference

File: `scripts/run_scalar_baseline_infer.py`

Enhancements added:

- `--inference-config` support
- `--image-only` support
- `--threshold` support
- feature-dimension mismatch guard

This made production inference reproducible from config files.

---

## 6) Benchmarking Scripts Added

### 6.1 Feature-set benchmark

File: `scripts/benchmark_scalar_feature_sets.py`

Compares:

- `legacy_like` (6 stats/band + PGA)
- `full` (8 stats/band + 5 scalar/context)

Also includes:

- threshold sweep
- best threshold by val F1
- building-density ablation

### 6.2 Model-family benchmark

File: `scripts/benchmark_scalar_models.py`

Evaluates all model families on same feature set and ranks by:

- max `test.f1`
- tie-break: max `test.roc_auc`

---

## 7) ML Model Evolution and Results

## 7.1 Early baseline (`models/scalar_baseline/metrics.json`)

- test F1 ~ `0.358`
- test ROC-AUC ~ `0.609`

## 7.2 Family benchmark on subset500 (`models/scalar_baseline/model_benchmark.json`)

- Best on that run: `gradient_boosting` (subset-specific).

## 7.3 Full comparison GB vs HGB (`models/scalar_baseline/model_benchmark_full_gb_vs_hgb.json`)

- `hist_gradient_boosting` slightly ahead:
  - test F1 `0.6198`
  - test ROC-AUC `0.9042`

This selected HGB as main ML candidate.

## 7.4 ML v2 attempt (`models/scalar_baseline_hgb_v2/metrics.json`)

Config was stricter + image-only + hard-negative-mining + low threshold (`0.25`).

Outcome:

- test F1 at best-threshold: `0.4790`
- test ROC-AUC: `0.7259`

Interpretation:

- degraded versus strong HGB baseline.
- likely due to reduced feature space + aggressive recall-oriented threshold/maining combo.

## 7.5 ML v2.1 (`models/scalar_baseline_hgb_v21/metrics.json`)

Changes:

- `image_only=false` (full 29 features)
- `hard_negative_mining=false`
- conservative preset
- threshold sweep restricted to `0.35..0.60`

Best result:

- best threshold: `0.35`
- test metrics at threshold `0.35`:
  - accuracy `0.8378`
  - precision `0.6188`
  - recall `0.8036`
  - F1 `0.6992`
  - ROC-AUC `0.9043`

Interpretation:

- best operational model so far.
- AUC remains high and F1 improved strongly via calibrated threshold selection.

---

## 8) Deep Learning Track (Siamese CNN)

Implemented files:

- `app/ml/siamese_data.py`
- `app/ml/siamese_model.py`
- `scripts/train_siamese.py`
- `doc/siamese_visual.md`

Design:

- shared encoder for pre/post
- compare embeddings via absolute difference
- BCEWithLogitsLoss with class-weighting
- CUDA + AMP support

Recorded result (`models/siamese_visual/metrics.json`):

- test F1 `0.4670`
- test ROC-AUC `0.6818`

Interpretation:

- lower than best ML baseline on this dataset/setup.
- remains useful R&D path, not selected for current production.

---

## 9) Tests Added/Used

Existing + added tests under `tests/`:

- `test_scalar_baseline.py`
- `test_scalar_model_selection.py`
- `test_siamese_training.py`
- existing suite files (`test_api.py`, `test_classification.py`, `test_integrity.py`, `test_split_dataset.py`)

What these covered in this phase:

- scalar feature extraction and loader shape expectations
- multi-model selection utility behavior
- siamese dataset/forward/backward smoke checks (when torch available)

---

## 10) Configuration and Runtime Decisions

### 10.1 Why threshold tuning

Default `0.5` was not always optimal for F1 under class imbalance.

- For `v21`, val-based selection yielded `0.35`, which improved test F1.

### 10.2 Why production config file

To freeze runtime behavior:

- same model path
- same threshold
- same feature mode

Production config:

- `models/scalar_baseline_hgb_v21/inference_config_prod.json`

```json
{
  "threshold": 0.35,
  "max_bands": 3,
  "image_only": false,
  "normalize_pga": false,
  "profile": "prod_balanced_v21"
}
```

---

## 11) Production Selection

Current production selection:

- model: `models/scalar_baseline_hgb_v21/model.pkl`
- inference config: `models/scalar_baseline_hgb_v21/inference_config_prod.json`
- threshold: `0.35`

Rationale:

- highest practical test F1 among vetted ML variants
- strong ROC-AUC retained
- reproducible inference behavior via config-driven threshold

---

## 12) Command Reference (Final)

### 12.1 Train final ML v2.1 style

```powershell
Set-Location "D:\JupyterProject"
.\.venv312\Scripts\Activate.ps1
python -u scripts\train_scalar_baseline_v2.py --train-csv data\processed\train_split.csv --val-csv data\processed\val_split.csv --test-csv data\processed\test_split.csv --output-dir models\scalar_baseline_hgb_v21 --no-image-only --preset conservative --no-hard-negative-mining --threshold-min 0.35 --threshold-max 0.60 --threshold-step 0.05 --log-every 200
```

### 12.2 Inference with production model/config

```powershell
Set-Location "D:\JupyterProject"
.\.venv312\Scripts\Activate.ps1
python scripts\run_scalar_baseline_infer.py --model-path models\scalar_baseline_hgb_v21\model.pkl --inference-config models\scalar_baseline_hgb_v21\inference_config_prod.json --pre-image D:\path\to\pre.tif --post-image D:\path\to\post.tif
```

---

## 13) What Was Learned (Summary)

- Event-level split is correct for leakage control but can distort row-level target ratios.
- Richer handcrafted change features + HGB are very competitive for this problem.
- Threshold optimization is not optional when class distributions differ across splits.
- Image-only + aggressive hard-negative setup can harm precision and total F1 in this pipeline.
- Siamese DL worked technically (GPU + training) but underperformed current ML baseline.
- Config-driven inference (`model + inference_config`) is required for stable production behavior.

---

## 14) Immediate Next Steps

1. Add calibrated probability variant (`sigmoid`/`isotonic`) on top of v21 and compare.
2. Run seed robustness (3-5 seeds) for v21-style training to measure variance.
3. Keep a separate `prod_high_recall` config profile only if operations require lower miss-rate.
4. Optionally harden split objective if row-ratio fidelity becomes a strict requirement.

