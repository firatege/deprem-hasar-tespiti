# Scalar PGA Baseline

This baseline trains a lightweight binary classifier using:

- pre/post GeoTIFF change statistics (mean/std + diff skewness/kurtosis per band)
- scalar seismic/context features:
  - `pga_value` (signed-log normalized by default)
  - `magnitude`
  - `depth_km`
  - `acquisition_delta_days`
  - `building_density` (or `building_count` fallback)

Policy mapping remains: `major-damage` + `destroyed` => `1`.

## Train

```powershell
Set-Location "D:\JupyterProject"
py scripts\train_scalar_baseline.py --train-csv data\processed\train_split.csv --val-csv data\processed\val_split.csv --test-csv data\processed\test_split.csv --output-dir models\scalar_baseline --model-type logreg --pga-default 0.0 --magnitude-default 0.0 --depth-km-default 0.0 --acquisition-delta-days-default 0.0 --building-density-default 0.0 --normalize-pga
```

Supported model types:

- `logreg`
- `random_forest`
- `extra_trees`
- `gradient_boosting`
- `hist_gradient_boosting`

Outputs:

- `models/scalar_baseline/model.pkl`
- `models/scalar_baseline/metrics.json`

## Single Inference

```powershell
Set-Location "D:\JupyterProject"
py scripts\run_scalar_baseline_infer.py --model-path models\scalar_baseline\model.pkl --pre-image storage\tiles\geotiffs\hold\images\guatemala-volcano_00000004_pre_disaster.tif --post-image storage\tiles\geotiffs\hold\images\guatemala-volcano_00000004_post_disaster.tif --pga-value 0.42 --magnitude 6.8 --depth-km 12.0 --acquisition-delta-days 4 --building-density 18 --normalize-pga
```

## Feature Set A/B Benchmark

`legacy_like`: 6 image stats per band + PGA only.

`full`: 8 image stats per band (including diff skewness/kurtosis) + PGA + magnitude + depth + acquisition delta + building density.

The benchmark also includes:

- threshold sweep (`0.30..0.70` by default) with best threshold by val F1
- building density ablation (`full` vs `full_no_building_density`)

```powershell
Set-Location "D:\JupyterProject"
py scripts\benchmark_scalar_feature_sets.py --train-csv data\processed\subset_500\train_split.csv --val-csv data\processed\subset_500\val_split.csv --test-csv data\processed\subset_500\test_split.csv --output-json models\scalar_baseline\feature_set_benchmark.json
```

Optional knobs:

- `--threshold-min 0.30 --threshold-max 0.70 --threshold-step 0.05`
- `--skip-building-density-ablation`

Output:

- `models/scalar_baseline/feature_set_benchmark.json`

## Model Family Benchmark (Same Features)

Run all model families on the same extracted feature set and pick the best product candidate.

```powershell
Set-Location "D:\JupyterProject"
py scripts\benchmark_scalar_models.py --train-csv data\processed\subset_500\train_split.csv --val-csv data\processed\subset_500\val_split.csv --test-csv data\processed\subset_500\test_split.csv --model-types logreg,random_forest,extra_trees,gradient_boosting,hist_gradient_boosting --output-json models\scalar_baseline\model_benchmark.json
```

Output:

- `models/scalar_baseline/model_benchmark.json`

## ML v2 (HGB + Threshold Tuning)

Trains `hist_gradient_boosting` with a compact hyperparameter grid and selects threshold on validation F1.

```powershell
Set-Location "D:\JupyterProject"
py scripts\train_scalar_baseline_v2.py --train-csv data\processed\train_split.csv --val-csv data\processed\val_split.csv --test-csv data\processed\test_split.csv --output-dir models\scalar_baseline_hgb_v2 --image-only
```

Recommended anti-overfit options (enabled by default):

- `--preset conservative`
- `--hard-negative-mining`
- `--hnm-threshold 0.6 --hnm-weight 3.0 --hnm-max-fraction 0.2`

Fast smoke run before full training:

```powershell
Set-Location "D:\JupyterProject"
py scripts\train_scalar_baseline_v2.py --train-csv data\processed\subset_500\train_split.csv --val-csv data\processed\subset_500\val_split.csv --test-csv data\processed\subset_500\test_split.csv --output-dir models\scalar_baseline_hgb_v2_smoke --image-only --max-candidates 6
```

Outputs:

- `models/scalar_baseline_hgb_v2/model.pkl`
- `models/scalar_baseline_hgb_v2/metrics.json`
- `models/scalar_baseline_hgb_v2/inference_config.json`

Inference (uses tuned threshold and image-only mode from config):

```powershell
Set-Location "D:\JupyterProject"
py scripts\run_scalar_baseline_infer.py --model-path models\scalar_baseline_hgb_v2\model.pkl --inference-config models\scalar_baseline_hgb_v2\inference_config.json --pre-image D:\path\to\pre.tif --post-image D:\path\to\post.tif
```

