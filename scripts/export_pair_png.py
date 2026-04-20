import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    # Use first 3 bands as RGB; if fewer, repeat first band.
    if arr.shape[0] >= 3:
        rgb = arr[:3]
    else:
        rgb = np.repeat(arr[:1], 3, axis=0)

    # Robust contrast stretch to make previews easier to inspect.
    lo = float(np.percentile(rgb, 2))
    hi = float(np.percentile(rgb, 98))
    scale = max(1e-6, hi - lo)
    x = np.clip((rgb - lo) / scale, 0.0, 1.0)
    return (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def _read_tif(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(out_dtype="float32")


def _save_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pre/post GeoTIFF pair as PNG previews (+ diff)")
    parser.add_argument("--pre-tif", type=Path, required=True)
    parser.add_argument("--post-tif", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tmp/maxar_trial"))
    parser.add_argument("--save-diff", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    pre_arr = _read_tif(args.pre_tif)
    post_arr = _read_tif(args.post_tif)

    pre_png = _to_rgb_uint8(pre_arr)
    post_png = _to_rgb_uint8(post_arr)

    pre_path = args.out_dir / "pre.png"
    post_path = args.out_dir / "post.png"
    _save_png(pre_png, pre_path)
    _save_png(post_png, post_path)

    print(f"saved: {pre_path.resolve()}")
    print(f"saved: {post_path.resolve()}")

    if args.save_diff:
        h = min(pre_png.shape[0], post_png.shape[0])
        w = min(pre_png.shape[1], post_png.shape[1])
        diff = np.abs(post_png[:h, :w].astype(np.int16) - pre_png[:h, :w].astype(np.int16)).astype(np.uint8)
        diff_path = args.out_dir / "diff.png"
        _save_png(diff, diff_path)
        print(f"saved: {diff_path.resolve()}")


if __name__ == "__main__":
    main()

