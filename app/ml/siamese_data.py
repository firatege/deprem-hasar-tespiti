import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SiamesePairDataset(Dataset):
    """Loads pre/post GeoTIFF pairs for binary damage classification."""

    def __init__(self, split_csv: Path, *, img_size: int = 256, max_bands: int = 3) -> None:
        self.split_csv = split_csv
        self.img_size = img_size
        self.max_bands = max_bands
        self.rows: list[tuple[Path, Path, int]] = []

        with split_csv.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"pre_image_path", "post_image_path", "binary_damage"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required columns in split CSV: {sorted(missing)}")

            for row in reader:
                self.rows.append(
                    (
                        Path(row["pre_image_path"]),
                        Path(row["post_image_path"]),
                        int(row["binary_damage"]),
                    )
                )

        if not self.rows:
            raise ValueError(f"No rows found in {split_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre_path, post_path, label = self.rows[idx]
        pre = self._load_and_prepare(pre_path)
        post = self._load_and_prepare(post_path)
        y = torch.tensor(float(label), dtype=torch.float32)
        return pre, post, y

    @property
    def labels(self) -> np.ndarray:
        return np.array([label for _, _, label in self.rows], dtype=np.int64)

    def _load_and_prepare(self, image_path: Path) -> torch.Tensor:
        with rasterio.open(image_path) as ds:
            arr = ds.read(out_dtype="float32")

        if arr.shape[0] >= self.max_bands:
            arr = arr[: self.max_bands]
        else:
            missing = self.max_bands - arr.shape[0]
            arr = np.concatenate([arr, np.zeros((missing, arr.shape[1], arr.shape[2]), dtype=np.float32)], axis=0)

        x = torch.from_numpy(arr)
        x = F.interpolate(x.unsqueeze(0), size=(self.img_size, self.img_size), mode="bilinear", align_corners=False).squeeze(0)

        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

