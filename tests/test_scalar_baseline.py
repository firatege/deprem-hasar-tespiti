import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.ml.scalar_baseline import compute_change_features, load_dataset_from_split


def _write_tif(path: Path, array: np.ndarray) -> None:
    bands, height, width = array.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        transform=from_origin(0, 0, 1, 1),
    ) as ds:
        ds.write(array.astype("float32"))


def test_compute_change_features_and_loader(tmp_path: Path) -> None:
    pre = np.stack([np.full((8, 8), 10), np.full((8, 8), 20), np.full((8, 8), 30)]).astype(np.float32)
    post = pre + 5.0

    pre_path = tmp_path / "pre.tif"
    post_path = tmp_path / "post.tif"
    _write_tif(pre_path, pre)
    _write_tif(post_path, post)

    features = compute_change_features(pre_path, post_path, max_bands=3)
    assert len(features) == 24
    assert all(np.isfinite(features))

    csv_path = tmp_path / "split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["event_id", "tile_id", "pre_image_path", "post_image_path", "binary_damage"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "ev1",
                "tile_id": "tile-1",
                "pre_image_path": str(pre_path),
                "post_image_path": str(post_path),
                "binary_damage": "1",
            }
        )

    x, y, meta = load_dataset_from_split(csv_path, pga_default=0.42, max_bands=3)
    assert x.shape == (1, 29)
    assert x[0, -5] == np.float32(np.log1p(0.42))
    assert y.tolist() == [1]
    assert meta[0]["pga_value"] == 0.42

