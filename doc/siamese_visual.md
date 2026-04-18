# Siamese Visual-Only Baseline

This baseline uses only:

- `pre_image_path`
- `post_image_path`
- `binary_damage`

No PGA or external scalar/context features are required.

## GPU Setup (RTX)

Install project dependencies first:

```powershell
Set-Location "D:\JupyterProject"
py -3 -m pip install -r requirements.txt
```

If CUDA is not detected after install, install a CUDA-enabled PyTorch wheel from the official selector for your driver/CUDA version:

- https://pytorch.org/get-started/locally/

Quick check:

```powershell
Set-Location "D:\JupyterProject"
py -3 -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
```

## Train (6GB VRAM friendly defaults)

```powershell
Set-Location "D:\JupyterProject"
py -3 -u scripts\train_siamese.py --train-csv data\processed\train_split.csv --val-csv data\processed\val_split.csv --test-csv data\processed\test_split.csv --output-dir models\siamese_visual --device auto --img-size 256 --batch-size 8 --grad-accum-steps 2 --epochs 8 --amp
```

Outputs:

- `models/siamese_visual/best.pt`
- `models/siamese_visual/metrics.json`

