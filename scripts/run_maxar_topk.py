import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import geopandas as gpd
import leafmap
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box, mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.scalar_baseline import build_scalar_features, compute_change_features


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def _find_key_column(pre_gdf, post_gdf):
    common = set(pre_gdf.columns).intersection(set(post_gdf.columns))
    preferred = ["quadkey", "tile_id", "tile", "grid:code", "grid_code", "utm_tile"]
    for col in preferred:
        if col in common:
            return col

    for col in sorted(common):
        low = col.lower()
        if "quad" in low or "tile" in low or "grid" in low:
            return col
    return None


def _pick_href_from_row(row, suffix: str = "") -> str:
    preferred = ["visual", "image", "url", "asset_href", "href", "browse"]
    for col in preferred:
        key = f"{col}{suffix}" if suffix else col
        if key in row.index:
            value = row[key]
            if isinstance(value, str) and value.startswith("http"):
                return value

    for value in row.values:
        if isinstance(value, str) and value.startswith("http"):
            return value
    raise RuntimeError("No downloadable URL found in selected row")


def _build_pairs(pre_gdf, post_gdf, max_pairs: int):
    pairs = []
    key_col = _find_key_column(pre_gdf, post_gdf)
    if key_col:
        merged = pre_gdf.merge(post_gdf, on=key_col, suffixes=("_pre", "_post"))
        for _, row in merged.iterrows():
            pairs.append(
                {
                    "method": f"key:{key_col}",
                    "pair_id": str(row[key_col]),
                    "pre_url": _pick_href_from_row(row, "_pre"),
                    "post_url": _pick_href_from_row(row, "_post"),
                }
            )
            if len(pairs) >= max_pairs:
                break
        return pairs

    if pre_gdf.crs is None or post_gdf.crs is None:
        raise RuntimeError("Missing CRS – cannot build spatial pairs")

    if pre_gdf.crs != post_gdf.crs:
        post_gdf = post_gdf.to_crs(pre_gdf.crs)

    joined = gpd.sjoin(pre_gdf, post_gdf, how="inner", predicate="intersects")
    if len(joined) == 0:
        raise RuntimeError("No intersecting pre/post geometries found")

    for _, row in joined.iterrows():
        pre_idx = row.name
        post_idx = int(row["index_right"])
        pre_row = pre_gdf.loc[pre_idx]
        post_row = post_gdf.loc[post_idx]
        pair_id = f"pre{pre_idx}_post{post_idx}"
        pairs.append(
            {
                "method": "spatial",
                "pair_id": pair_id,
                "pre_url": _pick_href_from_row(pre_row),
                "post_url": _pick_href_from_row(post_row),
            }
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def _reproject_to_match(src_path: Path, ref_path: Path, out_path: Path) -> Path:
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs

    with rasterio.open(src_path) as src:
        if src.crs == dst_crs:
            return src_path
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(crs=dst_crs, transform=transform, width=width, height=height)

        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return out_path


def _crop_to_common_extent(pre_tif: Path, post_tif: Path, out_dir: Path, tag: str):
    with rasterio.open(pre_tif) as pre_ds, rasterio.open(post_tif) as post_ds:
        pre_box = box(*pre_ds.bounds)
        post_box = box(*post_ds.bounds)
        common = pre_box.intersection(post_box)
        if common.is_empty:
            raise RuntimeError("No overlap between pre/post rasters")

        geom = [mapping(common)]
        outputs = {}
        for label, ds in [("pre", pre_ds), ("post", post_ds)]:
            out_image, out_transform = rio_mask(ds, geom, crop=True)
            meta = ds.meta.copy()
            meta.update(
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
            )
            out_path = out_dir / f"{tag}_{label}_cropped.tif"
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(out_image)
            outputs[label] = out_path

    return outputs["pre"], outputs["post"]


def _load_inference_config(path: Path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return {
        "threshold": float(cfg.get("threshold", 0.5)),
        "max_bands": int(cfg.get("max_bands", 3)),
        "image_only": bool(cfg.get("image_only", False)),
        "normalize_pga": bool(cfg.get("normalize_pga", True)),
    }


def _predict(model, inf_cfg, pre_path: Path, post_path: Path):
    x = compute_change_features(pre_path, post_path, max_bands=inf_cfg["max_bands"])
    if not inf_cfg["image_only"]:
        x.extend(
            build_scalar_features(
                pga_value=0.0,
                magnitude=0.0,
                depth_km=0.0,
                acquisition_delta_days=0.0,
                building_density=0.0,
                normalize_pga=inf_cfg["normalize_pga"],
            )
        )
    x_np = np.array([x], dtype=np.float32)
    prob = float(model.predict_proba(x_np)[0, 1])
    pred = int(prob >= inf_cfg["threshold"])
    return prob, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch score Maxar pre/post pairs and export top-k probabilities")
    parser.add_argument("--collection", default="Kahramanmaras-turkey-earthquake-23")
    parser.add_argument("--cutoff-date", default="2023-02-06")
    parser.add_argument("--out-dir", default="tmp/maxar_trial_batch")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=60)
    parser.add_argument("--out-csv", default="data/processed/maxar_top20.csv")
    parser.add_argument("--model-path", default="models/scalar_baseline_hgb_v21/model.pkl")
    parser.add_argument(
        "--inference-config",
        default="models/scalar_baseline_hgb_v21/inference_config_prod.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    _log(f"searching collection={args.collection} cutoff={args.cutoff_date}")
    pre_gdf = leafmap.maxar_search(collection=args.collection, end_date=args.cutoff_date)
    post_gdf = leafmap.maxar_search(collection=args.collection, start_date=args.cutoff_date)
    _log(f"search done pre_rows={len(pre_gdf)} post_rows={len(post_gdf)}")

    if len(pre_gdf) == 0 or len(post_gdf) == 0:
        raise RuntimeError("No pre/post items found")

    pairs = _build_pairs(pre_gdf, post_gdf, args.max_pairs)
    _log(f"candidate pairs={len(pairs)}")

    with Path(args.model_path).open("rb") as f:
        model = pickle.load(f)
    inf_cfg = _load_inference_config(Path(args.inference_config))

    results = []
    for i, pair in enumerate(pairs, start=1):
        tag = f"pair_{i:03d}"
        _log(f"[{i}/{len(pairs)}] scoring {pair['pair_id']}")
        try:
            pre_tif = out_dir / f"{tag}_pre.tif"
            post_tif = out_dir / f"{tag}_post.tif"

            t0 = time.perf_counter()
            leafmap.download_file(pair["pre_url"], str(pre_tif), overwrite=True)
            leafmap.download_file(pair["post_url"], str(post_tif), overwrite=True)
            dl_s = time.perf_counter() - t0

            post_tif = _reproject_to_match(post_tif, pre_tif, out_dir / f"{tag}_post_reproj.tif")
            pre_crop, post_crop = _crop_to_common_extent(pre_tif, post_tif, out_dir, tag)

            prob, pred = _predict(model, inf_cfg, pre_crop, post_crop)
            results.append(
                {
                    "rank": 0,
                    "pair_id": pair["pair_id"],
                    "method": pair["method"],
                    "probability": prob,
                    "threshold": inf_cfg["threshold"],
                    "binary_damage": pred,
                    "pre_url": pair["pre_url"],
                    "post_url": pair["post_url"],
                    "pre_cropped_tif": str(pre_crop.resolve()),
                    "post_cropped_tif": str(post_crop.resolve()),
                    "download_seconds": round(dl_s, 2),
                    "pre_size_mb": _fmt_mb(pre_tif),
                    "post_size_mb": _fmt_mb(post_tif),
                }
            )
        except Exception as exc:
            _log(f"skip {pair['pair_id']} reason={exc}")

    if not results:
        raise RuntimeError("No pairs were successfully scored")

    results.sort(key=lambda x: x["probability"], reverse=True)
    top = results[: args.top_k]
    for idx, row in enumerate(top, start=1):
        row["rank"] = idx

    fieldnames = [
        "rank",
        "pair_id",
        "method",
        "probability",
        "threshold",
        "binary_damage",
        "pre_url",
        "post_url",
        "pre_cropped_tif",
        "post_cropped_tif",
        "download_seconds",
        "pre_size_mb",
        "post_size_mb",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)

    _log(f"wrote top-{len(top)} CSV -> {out_csv.resolve()}")
    _log(
        f"best pair={top[0]['pair_id']} prob={top[0]['probability']:.4f} pred={top[0]['binary_damage']}"
    )


if __name__ == "__main__":
    main()

