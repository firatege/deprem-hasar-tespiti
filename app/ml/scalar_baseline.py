import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from scipy.ndimage import sobel


def _safe_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _skewness_and_kurtosis(values: np.ndarray) -> tuple[float, float]:
    flat = values.astype(np.float64, copy=False).ravel()
    if flat.size == 0:
        return 0.0, 0.0

    std = float(np.std(flat))
    if std <= 1e-12:
        return 0.0, 0.0

    mean = float(np.mean(flat))
    z = (flat - mean) / std
    skew = float(np.mean(z**3))
    kurtosis = float(np.mean(z**4) - 3.0)
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurtosis):
        kurtosis = 0.0
    return skew, kurtosis


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a = ((a - mu_a) ** 2).mean()
    var_b = ((b - mu_b) ** 2).mean()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    if abs(den) < 1e-12:
        return 1.0
    return float(np.clip(num / den, -1.0, 1.0))


def _edge_change(pre_band: np.ndarray, post_band: np.ndarray) -> float:
    pre_f = pre_band.astype(np.float64)
    post_f = post_band.astype(np.float64)
    pre_edge = np.hypot(sobel(pre_f, axis=0), sobel(pre_f, axis=1))
    post_edge = np.hypot(sobel(post_f, axis=0), sobel(post_f, axis=1))
    diff = np.abs(post_edge - pre_edge)
    pre_mag = pre_edge.mean()
    return float(diff.mean() / (pre_mag + 1e-6))


def _band_features(pre_band: np.ndarray, post_band: np.ndarray) -> list[float]:
    diff = np.abs(post_band - pre_band)
    diff_skew, diff_kurtosis = _skewness_and_kurtosis(diff)
    return [
        float(np.mean(pre_band)),
        float(np.mean(post_band)),
        float(np.mean(diff)),
        float(np.std(pre_band)),
        float(np.std(post_band)),
        float(np.std(diff)),
        diff_skew,
        diff_kurtosis,
        _ssim(pre_band, post_band),
        _edge_change(pre_band, post_band),
    ]


def build_scalar_features(
    *,
    pga_value: float,
    magnitude: float = 0.0,
    depth_km: float = 0.0,
    acquisition_delta_days: float = 0.0,
    building_density: float = 0.0,
    normalize_pga: bool = True,
) -> list[float]:
    pga_feature = _signed_log1p(pga_value) if normalize_pga else pga_value
    return [
        float(pga_feature),
        float(magnitude),
        float(depth_km),
        float(acquisition_delta_days),
        float(building_density),
    ]


def compute_change_features(pre_image_path: Path, post_image_path: Path, max_bands: int = 3) -> list[float]:
    """Extract compact per-band statistics from pre/post GeoTIFFs.

    This keeps training lightweight while still modeling temporal change.
    """
    with rasterio.open(pre_image_path) as pre_ds:
        pre = pre_ds.read(out_dtype="float32")
    with rasterio.open(post_image_path) as post_ds:
        post = post_ds.read(out_dtype="float32")

    shared_bands = min(pre.shape[0], post.shape[0], max_bands)
    if shared_bands == 0:
        return [0.0] * (max_bands * 8)

    features: list[float] = []
    for idx in range(shared_bands):
        features.extend(_band_features(pre[idx], post[idx]))

    # Pad when image has fewer than max_bands channels.
    missing = max_bands - shared_bands
    if missing > 0:
        features.extend([0.0] * (missing * 8))

    return features


def load_dataset_from_split(
    split_csv: Path,
    *,
    pga_default: float = 0.0,
    magnitude_default: float = 0.0,
    depth_km_default: float = 0.0,
    acquisition_delta_days_default: float = 0.0,
    building_density_default: float = 0.0,
    max_bands: int = 3,
    normalize_pga: bool = True,
    progress_every: int = 0,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    logger = log_fn or print

    rows_meta: list[dict[str, Any]] = []
    features: list[list[float]] = []
    labels: list[int] = []

    with split_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"tile_id", "pre_image_path", "post_image_path", "binary_damage"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in split CSV: {sorted(missing)}")

        for idx, row in enumerate(reader, start=1):
            pre_path = Path(row["pre_image_path"])
            post_path = Path(row["post_image_path"])
            y = int(row["binary_damage"])
            pga_raw = _safe_float(row.get("pga_value"), pga_default)
            magnitude = _safe_float(row.get("magnitude"), magnitude_default)
            depth_km = _safe_float(row.get("depth_km"), depth_km_default)
            acquisition_delta_days = _safe_float(
                row.get("acquisition_delta_days"), acquisition_delta_days_default
            )
            building_density = _safe_float(
                row.get("building_density"),
                _safe_float(row.get("building_count"), building_density_default),
            )

            x = compute_change_features(pre_path, post_path, max_bands=max_bands)
            x.extend(
                build_scalar_features(
                    pga_value=pga_raw,
                    magnitude=magnitude,
                    depth_km=depth_km,
                    acquisition_delta_days=acquisition_delta_days,
                    building_density=building_density,
                    normalize_pga=normalize_pga,
                )
            )

            if any(math.isnan(item) or math.isinf(item) for item in x):
                # Skip rare corrupt samples that lead to invalid stats.
                continue

            features.append(x)
            labels.append(y)
            rows_meta.append(
                {
                    "tile_id": row["tile_id"],
                    "event_id": row.get("event_id", ""),
                    "pga_value": pga_raw,
                    "magnitude": magnitude,
                    "depth_km": depth_km,
                    "acquisition_delta_days": acquisition_delta_days,
                    "building_density": building_density,
                }
            )

            if progress_every and (idx % progress_every == 0):
                logger(f"[loader] {split_csv.name}: processed={idx} valid={len(features)}")

    if not features:
        raise ValueError(f"No valid samples loaded from {split_csv}")

    if progress_every:
        logger(f"[loader] {split_csv.name}: done valid={len(features)}")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64), rows_meta

