import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


torch = pytest.importorskip("torch")

from app.ml.siamese_data import SiamesePairDataset
from app.ml.siamese_model import SiameseDamageNet
from scripts.train_siamese import resolve_device


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


def test_resolve_device_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_device("auto")
    assert str(device) == "cpu"


def test_siamese_dataset_and_forward_backward(tmp_path: Path) -> None:
    pre = np.stack([np.full((16, 16), 10), np.full((16, 16), 20), np.full((16, 16), 30)]).astype(np.float32)
    post = pre + 2.0

    pre_path = tmp_path / "pre.tif"
    post_path = tmp_path / "post.tif"
    _write_tif(pre_path, pre)
    _write_tif(post_path, post)

    csv_path = tmp_path / "split.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pre_image_path", "post_image_path", "binary_damage"])
        writer.writeheader()
        writer.writerow(
            {
                "pre_image_path": str(pre_path),
                "post_image_path": str(post_path),
                "binary_damage": "1",
            }
        )

    ds = SiamesePairDataset(csv_path, img_size=64, max_bands=3)
    pre_t, post_t, y = ds[0]
    assert pre_t.shape == (3, 64, 64)
    assert post_t.shape == (3, 64, 64)
    assert float(y.item()) == 1.0

    model = SiameseDamageNet(in_channels=3)
    logits = model(pre_t.unsqueeze(0), post_t.unsqueeze(0))
    assert logits.shape == (1,)

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, y.unsqueeze(0))
    loss.backward()
    assert float(loss.item()) >= 0.0

