# geo_sim/cli/landuse.py

from pathlib import Path
import numpy as np
import typer

import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
import shapely

import rasterio
from affine import Affine

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xarray as xr
import xrspatial as xs

from geo_sim.config.paths import SHP_LANDUSE_PATH, ALIGNED_TIFF_NIGHT_LIFE_OUT
from geo_sim.config.consts import _DPI, _GRID_STEP, _SUPER_K, _ALL_TOUCHED


# -----------------------------------------------------
# Visualization helpers
# -----------------------------------------------------


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
    im = plt.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
    plt.colorbar(im, label="landuse per cell (m)")
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
    out_png.parent.mkdir(parents=True, exist_ok=True)
    l, b, r, t = bounds

    fig, ax = plt.subplots()
    gdf.plot(ax=ax, linewidth=0.3, edgecolor="#1f77b4")
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    plt.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------
# Main processing
# -----------------------------------------------------


def landuse_feats() -> None:
    # -----------------------------
    # Validate paths
    # -----------------------------
    shp = Path(SHP_LANDUSE_PATH)
    ref_tif = Path(ALIGNED_TIFF_NIGHT_LIFE_OUT)

    if not shp.exists():
        raise typer.BadParameter(f"SHP_LANDUSE_PATH not found: {shp}")
    if not ref_tif.exists():
        raise typer.BadParameter(f"Reference aligned TIFF not found: {ref_tif}")

    # -----------------------------
    # Reference raster metadata
    # -----------------------------
    with rasterio.open(ref_tif) as ref:
        tgt_crs = ref.crs
        tgt_tx: Affine = ref.transform
        tgt_w, tgt_h = ref.width, ref.height

        ref_mask = ref.read_masks(1) > 0  # fast, small uint8 mask

        px_w = float(tgt_tx.a)
        px_h = float(-tgt_tx.e)
        pixel_area = abs(px_w * px_h)  # m^2

        ref_bounds = ref.bounds  # (left, bottom, right, top)

    if tgt_crs is None:
        raise typer.BadParameter("Reference TIFF has no CRS; cannot proceed.")

    # -----------------------------
    # Load & clean landuse geometries
    # -----------------------------
    gdf = gpd.read_file(shp)
    if gdf.empty:
        raise typer.BadParameter("Landuse shapefile has no features.")

    # keep non-empty geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    # reproject if needed
    if gdf.crs is not None and str(gdf.crs) != str(tgt_crs):
        gdf = gdf.to_crs(tgt_crs)
    elif gdf.crs is None:
        raise typer.BadParameter("Landuse shapefile has no CRS defined.")

    # clip to reference bounds
    clip_geom = box(*ref_bounds)
    gdf = gdf[gdf.intersects(clip_geom)]
    gdf = gdf.clip(gpd.GeoSeries([clip_geom], crs=tgt_crs))

    # explode once
    gdf = gdf.explode(index_parts=False, ignore_index=True)

    if gdf.empty:
        raise typer.BadParameter("No geometries intersect the reference extent.")

    # -----------------------------
    # Output paths
    # -----------------------------
    out_dir = ref_tif.parent
    png_outdir = out_dir / "pngs"
    png_outdir.mkdir(parents=True, exist_ok=True)

    out_vec_png = png_outdir / "landuses_vector_preview.png"
    out_tif = out_dir / "landuses_length_m.tif"
    out_png = png_outdir / "landuses_length_m.png"

    # -----------------------------
    # Vector preview
    # -----------------------------
    _save_vector_preview(
        gdf,
        out_vec_png,
        bounds=(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top),
        title="Landuses (vector preview)",
    )

    # -----------------------------
    # Buffering (stroke width)
    # -----------------------------
    k = max(2, int(_SUPER_K))
    fine_px_w = px_w / k
    fine_px_h = px_h / k
    stroke_width = min(fine_px_w, fine_px_h)
    buffer_dist = stroke_width / 2.0

    # Vectorized Shapely buffer (fast)
    gdf_buf = gdf.copy()
    gdf_buf["geometry"] = shapely.buffer(
        gdf_buf.geometry.values,
        buffer_dist,
        cap_style=2,
        join_style=2,
    )

    gdf_buf = gdf_buf[gdf_buf.geometry.notnull() & ~gdf_buf.geometry.is_empty].copy()

    # -----------------------------
    # Rasterize: accumulate polygon *area* per pixel
    # -----------------------------
    # Using xarray-spatial to rasterize geometries
    shapes = [
        (geom, geom.area) for geom in gdf_buf.geometry if isinstance(geom, BaseGeometry)
    ]

    # Convert to xarray
    da = xs.rasterize(
        shapes,
        out_shape=(tgt_h, tgt_w),
        transform=tgt_tx,
        fill=0.0,
        dtype="float32",
        merge_alg="add",
        all_touched=_ALL_TOUCHED,
    )

    area_sum = da.values
    area_fraction = area_sum / pixel_area

    # Convert covered area → length = area / stroke_width
    length_m = area_fraction * (pixel_area / max(stroke_width, 1e-9))

    # mask to reference raster validity
    length_m = np.where(ref_mask, length_m, np.nan).astype("float32")

    # -----------------------------
    # Write GeoTIFF
    # -----------------------------
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

    # -----------------------------
    # Quicklook raster
    # -----------------------------
    _save_quicklook(length_m, out_png, title="Landuse length per cell (m)")

    # -----------------------------
    # Report
    # -----------------------------
    finite = np.isfinite(length_m)
    total_len_km = float(np.nansum(length_m) / 1000.0)
    vmin = float(np.nanmin(length_m)) if np.any(finite) else float("nan")
    vmax = float(np.nanmax(length_m)) if np.any(finite) else float("nan")

    typer.echo("\nlanduses rasterization:")
    typer.echo(f"  ref grid        : {ref_tif}")
    typer.echo(f"  shp             : {shp}")
    typer.echo(f"  stroke_width    : {stroke_width:.3f} m")
    typer.echo(f"  out GeoTIFF     : {out_tif}")
    typer.echo(f"  out PNG (vector): {out_vec_png}")
    typer.echo(f"  out PNG (raster): {out_png}")
    typer.echo(f"  valid%          : {finite.mean() * 100:.1f}")
    typer.echo(f"  min/max (m)     : {vmin:.3g} / {vmax:.3g}")
    typer.echo(f"  total length    : ~{total_len_km:.1f} km")
    typer.echo("  ✓ Done")


if __name__ == "__main__":
    landuse_feats()
