# geo_sim/cli/buildings.py
from pathlib import Path
import pdb
import numpy as np
import typer

# deps
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry import base as shapely_base

import rasterio
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from affine import Affine
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_sim.config.paths import SHP_BUILDINGS_PATH, ALIGNED_TIFF_NIGHT_LIFE_OUT
from geo_sim.config.consts import _DPI, _GRID_STEP, _SUPER_K, _ALL_TOUCHED


def _overlay_grid(ax, width: int, height: int, step: int) -> None:
    if step <= 0:
        return
    x_minor = np.arange(-0.5, width - 0.5 + 1e-9, step)
    y_minor = np.arange(-0.5, height - 0.5 + 1e-9, step)
    ax.set_xticks(x_minor, minor=True)
    ax.set_yticks(y_minor, minor=True)
    ax.tick_params(which="both", labelbottom=False, labelleft=False, length=0)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(
        which="minor",
        linestyle="-",
        linewidth=0.25,
        alpha=0.35,
        color="white",
        zorder=10,
    )


def _save_quicklook(arr2d: np.ndarray, out_png: Path, title: str) -> None:
    data = arr2d.astype("float64")
    finite = np.isfinite(data)
    if np.any(finite):
        pos = data[finite & (data > 0)]
        if pos.size >= 50:
            vmin, vmax = 0.0, float(np.percentile(pos, 99.5))
        else:
            vmin = float(np.nanpercentile(data, 2))
            vmax = float(np.nanpercentile(data, 98))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nanmax(data))
            vmin = float(np.nanmin(data))
            if vmax <= vmin:
                vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e6e6e6")
    cmap.set_under("#202020")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    im = plt.imshow(data, vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, label="road length per cell (m)")
    plt.title(title)
    ax = plt.gca()
    _overlay_grid(ax, data.shape[1], data.shape[0], _GRID_STEP)
    plt.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close()


def _save_vector_preview(
    gdf: gpd.GeoDataFrame,
    out_png: Path,
    bounds: tuple[float, float, float, float],
    title: str,
) -> None:
    """Simple line plot of the (clipped, projected) buildings."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    l, b, r, t = bounds
    fig, ax = plt.subplots()
    # thin linewidth to avoid saturating dense areas
    gdf.plot(ax=ax, linewidth=0.3, edgecolor="#1f77b4")
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def buildings_feats() -> None:

    import geopandas as gpd
    import numpy as np
    from collections import Counter

    gdf = gpd.read_file(SHP_BUILDINGS_PATH)
    print("driver:", gdf._geometry_column_name)  # geometry col name
    print("CRS:", gdf.crs)
    print("rows:", len(gdf))

    geom_types = Counter(gdf.geometry.geom_type.fillna("None"))
    print("geom types:", geom_types)
    n_empty = int((gdf.geometry.is_empty | gdf.geometry.isna()).sum())
    print("empty geometries:", n_empty)

    # peek a few non-empty, if any
    non_empty = gdf[~(gdf.geometry.is_empty | gdf.geometry.isna())]
    print(non_empty.head(3))

    shp = Path(SHP_BUILDINGS_PATH)
    ref_tif = Path(ALIGNED_TIFF_NIGHT_LIFE_OUT)
    if not shp.exists():
        raise typer.BadParameter(f"path not found: {shp}")
    if not ref_tif.exists():
        raise typer.BadParameter(f"Reference aligned TIFF not found: {ref_tif}")

    # 1) reference grid
    with rasterio.open(ref_tif) as ref:
        tgt_crs = ref.crs
        tgt_tx: Affine = ref.transform
        tgt_w, tgt_h = ref.width, ref.height
        ref_mask = np.isfinite(ref.read(1, masked=True).filled(np.nan))
        # meters per pixel
        px_w = float(tgt_tx.a)
        px_h = float(-tgt_tx.e)
        coarse_area = abs(px_w * px_h)
        ref_bounds = ref.bounds  # (left, bottom, right, top)

    if tgt_crs is None:
        raise typer.BadParameter("Reference TIFF has no CRS; cannot proceed.")

    # 2) load buildings, fix/clip/project
    gdf = gpd.read_file(shp)
    if gdf.empty:
        raise typer.BadParameter(f"BUil shapefile has no features: {shp}")
    # keep non-empty geometries only
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()

    # explode multi-geom collections
    gdf = gdf.explode(index_parts=False, ignore_index=True)

    if gdf.crs is None:
        raise typer.BadParameter("Buildings shapefile has no CRS defined.")
    if str(gdf.crs) != str(tgt_crs):
        gdf = gdf.to_crs(tgt_crs)

    # clip to reference bounds to avoid plotting/rasterizing outside area
    clip_poly = gpd.GeoSeries(data=[box(*ref_bounds)], crs=tgt_crs)
    gdf = gdf.clip(clip_poly)
    gdf = gdf.explode(index_parts=False, ignore_index=True)

    if gdf.empty:
        raise typer.BadParameter("No line geometries found in the reference extent.")

    # OUTPUT paths
    out_dir = ref_tif.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    png_outdir = out_dir / "pngs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_vec_png = out_dir / png_outdir / "buildings_vector_preview.png"
    out_tif = out_dir / "buildings.tif"
    out_png = out_dir / png_outdir / "buildings.png"

    # 3) VECTOR PREVIEW (before rasterization)
    _save_vector_preview(
        gdf,
        out_vec_png,
        (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top),
        title="Buildings (vector preview)",
    )

    # 4) supersampled rasterization -> coverage fraction
    k = max(2, int(_SUPER_K))
    fine_tx = Affine(
        tgt_tx.a / k, tgt_tx.b / k, tgt_tx.c, tgt_tx.d / k, tgt_tx.e / k, tgt_tx.f
    )
    fine_w, fine_h = tgt_w * k, tgt_h * k

    fine_px_w = px_w / k
    fine_px_h = px_h / k
    stroke_width = min(fine_px_w, fine_px_h)
    buffer_dist = stroke_width / 2.0

    gdf_buf = gdf.copy()
    gdf_buf["geometry"] = gdf_buf.geometry.buffer(
        buffer_dist, cap_style=2, join_style=2
    )
    gdf_buf = gdf_buf[~gdf_buf.geometry.is_empty & gdf_buf.geometry.notnull()].copy()

    shapes = [
        (geom, 1)
        for geom in gdf_buf.geometry.values
        if isinstance(geom, shapely_base.BaseGeometry)
    ]
    fine = rasterize(
        shapes=shapes,
        out_shape=(fine_h, fine_w),
        transform=fine_tx,
        fill=0,
        dtype="uint8",
        all_touched=_ALL_TOUCHED,
        merge_alg=MergeAlg.replace,
    )
    fine = fine.astype("float32")
    cov = fine.reshape(tgt_h, k, tgt_w, k).mean(axis=(1, 3))

    # 5) convert coverage to length (m), mask to ref valid
    length_m = cov * coarse_area / max(stroke_width, 1e-9)
    length_m = np.where(ref_mask, length_m, np.nan).astype("float32")

    # 6) write aligned GeoTIFF
    profile = {
        "driver": "GTiff",
        "height": tgt_h,
        "width": tgt_w,
        "count": 1,
        "dtype": "float32",
        "crs": tgt_crs,
        "transform": tgt_tx,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
        "interleave": "band",
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(length_m, 1)

    # 7) RASTER QUICKLOOK (after)
    _save_quicklook(length_m, out_png, title="Road length per cell (m) — raster")

    # 8) report
    finite = np.isfinite(length_m)
    total_len_km = float(np.nansum(length_m) / 1000.0)
    vmin = float(np.nanmin(length_m)) if np.any(finite) else float("nan")
    vmax = float(np.nanmax(length_m)) if np.any(finite) else float("nan")
    typer.echo("\nBuildings rasterization:")
    typer.echo(f"  ref grid        : {ref_tif}")
    typer.echo(f"  shp             : {shp}")
    typer.echo(f"  k (supersample) : {k}  stroke_width ≈ {stroke_width:.3f} m")
    typer.echo(f"  out GeoTIFF     : {out_tif}")
    typer.echo(f"  out PNG (vector): {out_vec_png}")
    typer.echo(f"  out PNG (raster): {out_png}")
    typer.echo(f"  valid%          : {finite.mean()*100:.1f}")
    typer.echo(f"  min/max (m)     : {vmin:.3g} / {vmax:.3g}")
    typer.echo(f"  total length    : ~{total_len_km:.1f} km (sum of per-cell est.)")
    typer.echo("  ✓ Done")


# wire into CLI (in app.py):
# from .buildings import buildings_to_length
# app.command("buildings-length")(buildings_to_length)

if __name__ == "__main__":
    buildings_feats()
