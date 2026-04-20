from pathlib import Path
import rasterio

pre = Path(r"D:\JupyterProject\tmp\maxar_trial\pre.tif")
post = Path(r"D:\JupyterProject\tmp\maxar_trial\post.tif")

with rasterio.open(pre) as a, rasterio.open(post) as b:
    print("PRE  crs:", a.crs, "shape:", (a.height, a.width), "bounds:", a.bounds)
    print("POST crs:", b.crs, "shape:", (b.height, b.width), "bounds:", b.bounds)

    same_crs = a.crs == b.crs
    same_shape = (a.height, a.width) == (b.height, b.width)

    # Basit overlap oranı
    left = max(a.bounds.left, b.bounds.left)
    right = min(a.bounds.right, b.bounds.right)
    bottom = max(a.bounds.bottom, b.bounds.bottom)
    top = min(a.bounds.top, b.bounds.top)
    inter = max(0, right - left) * max(0, top - bottom)
    area_a = (a.bounds.right - a.bounds.left) * (a.bounds.top - a.bounds.bottom)
    area_b = (b.bounds.right - b.bounds.left) * (b.bounds.top - b.bounds.bottom)
    overlap = inter / max(1e-9, min(area_a, area_b))

    print("same_crs:", same_crs)
    print("same_shape:", same_shape)
    print("overlap_ratio:", round(overlap, 4))