import argparse
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import leafmap
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping


def _log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}")


def _fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def _find_key_column(pre_gdf, post_gdf):
    """Find a shared tile/grid column for deterministic pairing."""
    common = set(pre_gdf.columns).intersection(set(post_gdf.columns))
    preferred = [
        "quadkey",
        "tile_id",
        "tile",
        "grid:code",
        "grid_code",
        "utm_tile",
    ]
    for col in preferred:
        if col in common:
            return col

    for col in sorted(common):
        low = col.lower()
        if "quad" in low or "tile" in low or "grid" in low:
            return col
    return None


def _pick_href_from_row(row, suffix="") -> str:
    """Extract the first downloadable URL from a GeoDataFrame row.

    When the row comes from a merge with suffixes (_pre / _post),
    pass suffix="_pre" or suffix="_post" so the preferred column
    names are adjusted accordingly.
    """
    preferred = ["visual", "image", "url", "asset_href", "href", "browse"]
    for col in preferred:
        check = f"{col}{suffix}" if suffix else col
        if check in row.index:
            value = row[check]
            if isinstance(value, str) and value.startswith("http"):
                return value

    # fallback: grab the first http string in the row
    for value in row.values:
        if isinstance(value, str) and value.startswith("http"):
            return value
    raise RuntimeError("No downloadable URL found in selected row")


def _choose_best_pair(pre_gdf, post_gdf):
    """Return the best matching pre/post pair, either by key or spatially."""

    key_col = _find_key_column(pre_gdf, post_gdf)
    if key_col:
        merged = pre_gdf.merge(post_gdf, on=key_col, suffixes=("_pre", "_post"))
        if len(merged) > 0:
            row = merged.iloc[0]
            return {
                "method": f"key:{key_col}",
                "key_value": row[key_col],
                "pre_url": _pick_href_from_row(row, suffix="_pre"),
                "post_url": _pick_href_from_row(row, suffix="_post"),
            }

    # fallback: spatial intersection
    if pre_gdf.crs is None or post_gdf.crs is None:
        raise RuntimeError("Missing CRS – cannot spatially pair pre/post")

    if pre_gdf.crs != post_gdf.crs:
        post_gdf = post_gdf.to_crs(pre_gdf.crs)

    joined = gpd.sjoin(pre_gdf, post_gdf, how="inner", predicate="intersects")
    if len(joined) == 0:
        raise RuntimeError("No intersecting pre/post geometries found")

    best = None
    for _, row in joined.iterrows():
        pre_idx = row.name
        post_idx = int(row["index_right"])
        pre_geom = pre_gdf.loc[pre_idx].geometry
        post_geom = post_gdf.loc[post_idx].geometry
        inter = pre_geom.intersection(post_geom).area
        denom = max(1e-9, min(pre_geom.area, post_geom.area))
        overlap = float(inter / denom)
        if best is None or overlap > best["overlap"]:
            best = {
                "pre_idx": pre_idx,
                "post_idx": post_idx,
                "overlap": overlap,
            }

    pre_row = pre_gdf.loc[best["pre_idx"]]
    post_row = post_gdf.loc[best["post_idx"]]

    return {
        "method": f"spatial (overlap={best['overlap']:.2%})",
        "key_value": None,
        "pre_url": _pick_href_from_row(pre_row),
        "post_url": _pick_href_from_row(post_row),
    }


def _reproject_to_match(src_path: Path, ref_path: Path, out_path: Path) -> None:
    """Reproject src_path to match the CRS of ref_path, write to out_path."""
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs

    with rasterio.open(src_path) as src:
        if src.crs == dst_crs:
            return  # nothing to do
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
    _log(f"reprojected {src_path.name} → {dst_crs}")


def _crop_to_common_extent(pre_tif: Path, post_tif: Path, out_dir: Path):
    """Crop both rasters to their overlapping bounding box.

    Returns (cropped_pre_path, cropped_post_path).
    """
    with rasterio.open(pre_tif) as pre_ds, rasterio.open(post_tif) as post_ds:
        pre_box = box(*pre_ds.bounds)
        post_box = box(*post_ds.bounds)
        common = pre_box.intersection(post_box)

        if common.is_empty:
            raise RuntimeError(
                "Pre and post rasters do not overlap!\n"
                f"  pre bounds  = {pre_ds.bounds}\n"
                f"  post bounds = {post_ds.bounds}"
            )

        overlap_pct = common.area / min(pre_box.area, post_box.area) * 100
        _log(f"common extent overlap = {overlap_pct:.1f}%")

        geom = [mapping(common)]
        results = {}
        for label, ds, src_path in [("pre", pre_ds, pre_tif), ("post", post_ds, post_tif)]:
            out_image, out_transform = rio_mask(ds, geom, crop=True)
            meta = ds.meta.copy()
            meta.update(
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
            )
            cropped = out_dir / f"{label}_cropped.tif"
            with rasterio.open(cropped, "w", **meta) as dst:
                dst.write(out_image)
            results[label] = cropped
            _log(f"{label} cropped → {out_image.shape[1]}x{out_image.shape[2]}, {_fmt_mb(cropped)}")

    return results["pre"], results["post"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download matching Maxar pre/post tiles and run inference"
    )
    parser.add_argument("--collection", default="Kahramanmaras-turkey-earthquake-23")
    parser.add_argument("--cutoff-date", default="2023-02-06")
    parser.add_argument("--out-dir", default="tmp/maxar_trial")
    parser.add_argument("--model-path", default="models/scalar_baseline_hgb_v21/model.pkl")
    parser.add_argument(
        "--inference-config",
        default="models/scalar_baseline_hgb_v21/inference_config_prod.json",
    )
    parser.add_argument("--skip-inference", action="store_true", help="Download only, skip model run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Search ───────────────────────────────────────────────
    _log(f"[1/5] searching collection={args.collection} cutoff={args.cutoff_date}")
    pre_gdf = leafmap.maxar_search(collection=args.collection, end_date=args.cutoff_date)
    post_gdf = leafmap.maxar_search(collection=args.collection, start_date=args.cutoff_date)
    _log(f"found pre={len(pre_gdf)}  post={len(post_gdf)}")

    if len(pre_gdf) == 0 or len(post_gdf) == 0:
        _log("columns available:")
        _log(f"  pre:  {pre_gdf.columns.tolist()}")
        _log(f"  post: {post_gdf.columns.tolist()}")
        raise RuntimeError("No pre/post items found. Check collection name and date.")

    # ── 2. Pair ─────────────────────────────────────────────────
    _log("[2/5] selecting best pre/post pair")
    pair = _choose_best_pair(pre_gdf, post_gdf)
    _log(f"pair method={pair['method']}  key={pair['key_value']}")

    pre_tif = out_dir / "pre.tif"
    post_tif = out_dir / "post.tif"

    # ── 3. Download ─────────────────────────────────────────────
    _log("[3/5] downloading pre")
    t0 = time.perf_counter()
    leafmap.download_file(pair["pre_url"], str(pre_tif), overwrite=True)
    _log(f"pre done {time.perf_counter() - t0:.1f}s  {_fmt_mb(pre_tif)}")

    _log("downloading post")
    t0 = time.perf_counter()
    leafmap.download_file(pair["post_url"], str(post_tif), overwrite=True)
    _log(f"post done {time.perf_counter() - t0:.1f}s  {_fmt_mb(post_tif)}")

    if not pre_tif.exists() or not post_tif.exists():
        raise RuntimeError("Download failed")

    # ── 4. Align & crop ────────────────────────────────────────
    _log("[4/5] aligning rasters")
    with rasterio.open(pre_tif) as pds, rasterio.open(post_tif) as pods:
        _log(f"pre  crs={pds.crs} shape=({pds.height},{pds.width})")
        _log(f"post crs={pods.crs} shape=({pods.height},{pods.width})")
        crs_match = pds.crs == pods.crs

    if not crs_match:
        _log("CRS mismatch – reprojecting post to match pre")
        reproj_path = out_dir / "post_reproj.tif"
        _reproject_to_match(post_tif, pre_tif, reproj_path)
        post_tif = reproj_path

    pre_cropped, post_cropped = _crop_to_common_extent(pre_tif, post_tif, out_dir)

    # ── 5. Inference ────────────────────────────────────────────
    if args.skip_inference:
        _log("skipping inference (--skip-inference)")
        _log(f"ready:\n  pre  = {pre_cropped}\n  post = {post_cropped}")
    else:
        _log("[5/5] running production inference")
        cmd = [
            sys.executable,
            "scripts/run_scalar_baseline_infer.py",
            "--model-path",
            args.model_path,
            "--inference-config",
            args.inference_config,
            "--pre-image",
            str(pre_cropped.resolve()),
            "--post-image",
            str(post_cropped.resolve()),
        ]
        subprocess.run(cmd, check=True)

    _log("done")


if __name__ == "__main__":
    main()