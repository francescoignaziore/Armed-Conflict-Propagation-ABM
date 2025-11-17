# geo_sim/cli/batch_downsample.py
from pathlib import Path
import pdb
from typing import Tuple
import numpy as np
import typer
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    aligned_target,
)
from pyproj import CRS, Transformer

# matplotlib (headless)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geo_sim.config.paths import TIFF_PATHS, TIFF_OUT_DIR  # adjust names if needed
from geo_sim.config.consts import (
    OUTPUT_CRS_NIGHT_LIGHTS,
    OUTPUT_AREA_CELL,
    _DPI,
    _GRID_STEP,
)


def _overlay_grid(ax, width: int, height: int, step: int) -> None:
    """Pixel-aligned grid: lines at pixel boundaries every `step` pixels."""
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
        which="minor", linestyle="-", linewidth=0.3, alpha=0.6, color="white", zorder=10
    )


def _robust_vmin_vmax(data: np.ndarray) -> tuple[float, float]:
    """Choose sane (vmin, vmax) for sparse/zero-heavy rasters."""
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 1.0

    # Positive-only stats if we have enough positives (common for night-lights)
    pos = data[finite & (data > 0)]
    if pos.size >= 50:
        vmin = 0.0
        vmax = float(np.percentile(pos, 99.5))
    else:
        vmin = float(np.nanpercentile(data, 2))
        vmax = float(np.nanpercentile(data, 98))

    # Fallbacks if percentiles collapse
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            # final nudge to avoid vmin==vmax
            eps = 1.0 if vmin == 0 else max(1e-6, abs(vmin) * 0.01)
            vmax = vmin + eps
    return vmin, vmax


def _save_quicklook(arr2d: np.ndarray, out_png: Path, title: str) -> None:
    """Save a quicklook PNG with robust contrast (NaN-safe), nearest interp."""
    data = arr2d.astype("float64")
    vmin, vmax = _robust_vmin_vmax(data)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    # (optional) show NaNs as light gray instead of white
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#e6e6e6")

    im = plt.imshow(data, vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
    plt.colorbar(im, label="value")
    plt.title(title)
    ax = plt.gca()
    _overlay_grid(ax, width=data.shape[1], height=data.shape[0], step=_GRID_STEP)
    plt.savefig(out_png, dpi=_DPI, bbox_inches="tight")
    plt.close()


def _describe_raster(path: Path) -> str:
    lines = []
    with rasterio.open(path) as ds:
        left, bottom, right, top = ds.bounds
        lines.append("─ File")
        lines.append(f"  path        : {path}")
        lines.append(f"  driver      : {ds.driver}")
        lines.append(f"  size        : {ds.width} x {ds.height} (W x H)")
        lines.append(f"  count       : {ds.count} band(s)")
        lines.append(f"  dtype       : {', '.join(map(str, ds.dtypes))}")
        lines.append(f"  crs         : {ds.crs}")
        lines.append(f"  transform   : {ds.transform}")
        lines.append(f"  res (x,y)   : {ds.res[0]} , {ds.res[1]}")
        lines.append(
            f"  bounds      : left={left}, bottom={bottom}, right={right}, top={top}"
        )
        try:
            lines.append(
                f"  tiled       : {getattr(ds, 'tiled', None)}  blockshape={ds.block_shapes[0]}"
            )
        except Exception:
            pass
        arr = ds.read(1, masked=True, out_dtype="float64")
        data = arr.filled(np.nan)
        finite = np.isfinite(data)
        if np.any(finite):
            vmin = float(np.nanmin(data))
            vmax = float(np.nanmax(data))
            p2 = float(np.nanpercentile(data, 2))
            p98 = float(np.nanpercentile(data, 98))
            lines.append("─ Band 1 stats (nodata ignored)")
            lines.append(f"  valid px    : {int(finite.sum())}")
            lines.append(f"  min / max   : {vmin:.6g} / {vmax:.6g}")
            lines.append(f"  p2 / p98    : {p2:.6g} / {p98:.6g}")
        else:
            lines.append("─ Band 1 stats: no valid pixels")
    return "\n".join(lines)


def _reproject_bounds(
    bounds: Tuple[float, float, float, float], src_crs: CRS, dst_crs: CRS
) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    tfm = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xs = [left, right, left, right]
    ys = [bottom, bottom, top, top]
    X, Y = tfm.transform(xs, ys)
    return (min(X), min(Y), max(X), max(Y))


def tiff_alignment() -> None:
    if not TIFF_PATHS:
        raise typer.BadParameter("TIFF_PATHS is empty.")

    TIFF_OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_OUT_DIR = TIFF_OUT_DIR / "pngs"
    target_crs = CRS.from_user_input(OUTPUT_CRS_NIGHT_LIGHTS)
    target_side = float(OUTPUT_AREA_CELL) ** 0.5  # meters per pixel

    # 1) Verify common source CRS, print info, collect union bounds in target CRS
    first_crs = None
    union_left = +np.inf
    union_bottom = +np.inf
    union_right = -np.inf
    union_top = -np.inf

    typer.echo(
        f"Target CRS: {target_crs.to_string()}   target pixel ~ {target_side:.3f} m"
    )

    for p in TIFF_PATHS:
        if not p.exists():
            raise typer.BadParameter(f"Missing file: {p}")
        typer.echo(_describe_raster(p))
        with rasterio.open(p) as ds:
            if first_crs is None:
                first_crs = ds.crs
            elif ds.crs != first_crs:
                raise typer.BadParameter(
                    f"CRS mismatch: {p} has {ds.crs}, expected {first_crs}"
                )

            l, b, r, t = _reproject_bounds(ds.bounds, ds.crs, target_crs)
            union_left = min(union_left, l)
            union_bottom = min(union_bottom, b)
            union_right = max(union_right, r)
            union_top = max(union_top, t)

    # 2) Build a reference grid once (union bounds) and snap to resolution
    width_guess = max(1, int((union_right - union_left) / target_side))
    height_guess = max(1, int((union_top - union_bottom) / target_side))
    init_tx, _, _ = calculate_default_transform(
        target_crs,
        target_crs,
        width_guess,
        height_guess,
        left=union_left,
        bottom=union_bottom,
        right=union_right,
        top=union_top,
        resolution=(target_side, target_side),
    )
    tgt_tx, tgt_w, tgt_h = aligned_target(
        init_tx, width_guess, height_guess, (target_side, target_side)
    )

    cell_area = abs(tgt_tx.a * tgt_tx.e)
    typer.echo("\nReference grid:")
    typer.echo(
        f"  bounds (target) : left={union_left:.3f}, bottom={union_bottom:.3f}, right={union_right:.3f}, top={union_top:.3f}"
    )
    typer.echo(f"  transform       : {tgt_tx}")
    typer.echo(f"  shape (H x W)   : {tgt_h} x {tgt_w}")
    typer.echo(
        f"  pixel (m)       : {tgt_tx.a:.3f} × {abs(tgt_tx.e):.3f}  (area {abs(cell_area):.3f} m²)\n"
    )

    # 3) Process each input: scale/offset, reproject+downsample to shared grid, write TIFF + PNG
    for p in TIFF_PATHS:
        with rasterio.open(p) as src:
            # read as float with mask
            arr = src.read(1, masked=True, out_dtype="float32")
            data = arr.filled(np.nan)

            # radiometric transform
            scale = (
                src.scales[0] if src.scales and src.scales[0] not in (None, 1) else 1.0
            )
            offset = (
                src.offsets[0]
                if src.offsets and src.offsets[0] not in (None, 0)
                else 0.0
            )
            data = data * float(scale) + float(offset)

            # destination array (aligned to reference grid)
            dst = np.full((1, tgt_h, tgt_w), np.nan, dtype=np.float32)
            reproject(
                source=data,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tgt_tx,
                dst_crs=target_crs,
                resampling=Resampling.average,  # choose per layer if needed
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )

        # write GeoTIFF
        out_name = p.stem + f"_aligned_{int(round(target_side))}m.tif"
        out_path = TIFF_OUT_DIR / out_name
        profile = {
            "driver": "GTiff",
            "height": tgt_h,
            "width": tgt_w,
            "count": 1,
            "dtype": "float32",
            "crs": target_crs,
            "transform": tgt_tx,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": np.nan,
            "interleave": "band",
        }
        with rasterio.open(out_path, "w", **profile) as dst_ds:
            if "viirs" in str(src):
                dst = np.log(np.log(np.log(dst + 1) + 1) + 1)
                dst = np.where(dst < 0.75, 0, dst)
            dst_ds.write(dst)

        # save quicklook PNG
        png_path = PNG_OUT_DIR / (out_path.stem + f"_gridlev_{_GRID_STEP}_" + ".png")

        _save_quicklook(
            dst[0], png_path, title=f"{p.stem} — aligned {int(round(target_side))} m"
        )

        # quick log
        finite = np.isfinite(dst[0])
        vmin = float(np.nanmin(dst[0])) if np.any(finite) else float("nan")
        vmax = float(np.nanmax(dst[0])) if np.any(finite) else float("nan")
        typer.echo(
            f"✓ {p.name} → {out_path.name}  (valid%={finite.mean()*100:.1f}  min={vmin:.3g}  max={vmax:.3g})"
        )
        typer.echo(f"  PNG: {png_path}")

    typer.echo(
        "\nAll rasters written and quicklooks saved, aligned on the shared grid."
    )


# Optional: wire into your CLI app
if __name__ == "__main__":
    tiff_alignment()
