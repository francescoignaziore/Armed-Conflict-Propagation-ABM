import pdb
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import rasterio

from geo_sim.config.paths import TIFF_OUT_DIR
from geo_sim.config.consts import GEO_FEATURES_DISTRIBUTION

import matplotlib

matplotlib.use("Agg")  # HPC / headless
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers: normalization & plotting
# ---------------------------------------------------------------------------


def _normalize_01_with_nodata(arr: np.ndarray, nodata) -> np.ndarray:
    """
    Normalize a raster to [0, 1], ignoring nodata and non-finite values.
    Nodata and non-finite are returned as NaN.
    """
    arr = arr.astype(float)
    mask_invalid = ~np.isfinite(arr)
    if nodata is not None:
        mask_invalid |= arr == nodata

    vals = arr[~mask_invalid]
    if vals.size == 0:
        # Entire raster is invalid → all NaN
        out = np.full_like(arr, np.nan, dtype=float)
        return out

    vmin = float(vals.min())
    vmax = float(vals.max())
    out = arr.copy()

    if vmax > vmin:
        out[~mask_invalid] = (out[~mask_invalid] - vmin) / (vmax - vmin)
    else:
        # Constant (non-NaN) array
        out[~mask_invalid] = 0.0

    out[mask_invalid] = np.nan
    return out


def _plot_array(arr: np.ndarray, name: str, out_dir: Path, clip_percent: float = 99.5):
    """
    Save a PNG image of the (already normalized) raster array with percentile clipping.
    """
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        print(f"[{name}] No finite values to plot.")
        return

    low = (100.0 - clip_percent) / 2.0
    high = 100.0 - low
    vmin, vmax = np.percentile(vals, [low, high])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"

    plt.figure(figsize=(6, 5))
    plt.title(name)
    plt.imshow(arr, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(label="normalized value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[DEBUG] {name}: saved image plot to {out_path}")


def _plot_unique_hist(arr: np.ndarray, name: str, out_dir: Path, max_unique: int = 50):
    """
    Save a bar plot where each bar corresponds to a unique value in the array.
    Works on the (already normalized) array.
    If there are more than `max_unique` unique values, keep only the most frequent ones.
    """
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        print(f"[{name}] No finite values for unique-value histogram.")
        return

    uniques, counts = np.unique(vals, return_counts=True)

    if uniques.size > max_unique:
        idx = np.argsort(counts)[::-1][:max_unique]  # most frequent first
        uniques = uniques[idx]
        counts = counts[idx]

        # Sort by value for nicer x-axis
        order = np.argsort(uniques)
        uniques = uniques[order]
        counts = counts[order]

        print(
            f"[{name}] Unique values truncated to top {max_unique} by frequency "
            f"(original uniques: {len(np.unique(vals))})."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_unique_hist.png"

    plt.figure(figsize=(8, 4))
    plt.title(f"{name} – Unique normalized values (up to {max_unique})")
    plt.bar(range(len(uniques)), counts)
    plt.xticks(
        ticks=range(len(uniques)),
        labels=[f"{u:.3g}" for u in uniques],
        rotation=90,
    )
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[DEBUG] {name}: saved unique-value histogram to {out_path}")


# ---------------------------------------------------------------------------
# Stack rasters (filtered by GEO_FEATURES_DISTRIBUTION) and normalize
# ---------------------------------------------------------------------------


def stack_rasters(paths: Iterable[Path]) -> Tuple[np.ndarray, List[str]]:
    """
    Load single-band rasters, filter by GEO_FEATURES_DISTRIBUTION, normalize each
    to [0, 1] (nodata -> NaN), produce debug plots, and stack into (M, H, W).
    """
    layers: List[np.ndarray] = []
    names: List[str] = []

    debug_dir = TIFF_OUT_DIR / "simulation_plots"

    for p in paths:
        name = p.stem

        # Keep only rasters whose stem appears in any GEO_FEATURES_DISTRIBUTION entry
        if not any(feature in name for feature in GEO_FEATURES_DISTRIBUTION):
            continue

        with rasterio.open(p) as ds:
            arr = ds.read(1)  # (H, W)
            nodata = ds.nodata

        # Normalize to [0, 1], nodata -> NaN
        norm_arr = _normalize_01_with_nodata(arr, nodata)

        # Debug plots of normalized data
        _plot_array(norm_arr, name=name, out_dir=debug_dir)
        # _plot_unique_hist(norm_arr, name=name, out_dir=debug_dir)

        layers.append(norm_arr)
        names.append(name)

    if not layers:
        raise ValueError("No rasters matched GEO_FEATURES_DISTRIBUTION.")

    tensor = np.stack(layers, axis=0)  # (M, H, W)
    return tensor, names


# ---------------------------------------------------------------------------
# Sampling over grid cells
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sampling over grid cells (exp-based AND uniform)
# ---------------------------------------------------------------------------


def sample_cells(
    tensor: np.ndarray,
    N: int,
    *,
    grid: Optional[np.ndarray] = None,
    out_dir: Optional[Path] = None,
    base_name: str = "samples_overlay",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample N grid cells based on two distributions:

    1) Exp-based distribution:
       - For each cell (h, w), let v_m = tensor[m, h, w].
       - If ALL v_m are 0 or NaN -> probability 0 for that cell.
       - Otherwise total(h, w) = sum_m v_m (ignoring NaNs).
         p_exp(h, w) ∝ exp(total(h, w)).

    2) Uniform distribution over the SAME support (same active cells):
       - p_uni(h, w) = 1 / (#active_cells) for active cells, 0 otherwise.

    If `grid` is provided, it must be a (H, W) mask (bool or 0/1) and further
    restricts the support.

    If `out_dir` is provided:
      - Saves an overlay of exp-based samples on background.
      - Saves an overlay of uniform samples on background.
      - Saves a heatmap of the exp-based probability distribution.

    Returns:
        rows_exp, cols_exp, rows_uni, cols_uni
    """
    # tensor: (M, H, W)
    if tensor.ndim != 3:
        raise ValueError(f"tensor must be (M, H, W), got shape {tensor.shape}")

    M, H, W = tensor.shape

    # Identify cells where all rasters are 0 or NaN
    is_nan = np.isnan(tensor)
    is_zero = np.isfinite(tensor) & (tensor == 0.0)
    all_zero_or_nan = np.all(is_nan | is_zero, axis=0)  # (H, W)

    # Sum over rasters, ignoring NaNs
    total = np.nansum(tensor, axis=0)  # (H, W); NaNs treated as 0 in sum

    # Active support: cells that CAN get probability
    active_mask = ~all_zero_or_nan

    # Apply optional grid mask (further restriction)
    if grid is not None:
        if grid.shape != (H, W):
            raise ValueError(
                f"grid shape {grid.shape} does not match (H, W) = {(H, W)}"
            )
        active_mask &= grid.astype(bool)

    if not np.any(active_mask):
        raise ValueError("No active cells to sample from (support is empty).")

    # ---------------------------
    # 1) Exp-based probabilities
    # ---------------------------
    weights_exp = np.zeros((H, W), dtype=float)
    weights_exp[active_mask] = np.exp(total[active_mask])

    flat_weights_exp = weights_exp.ravel()
    total_weight_exp = flat_weights_exp.sum()
    if total_weight_exp <= 0:
        raise ValueError(
            "Total exp-based weight is zero; cannot sample. Check your data."
        )

    probs_exp = flat_weights_exp / total_weight_exp

    # ---------------------------
    # 2) Uniform probabilities over same support
    # ---------------------------
    weights_uni = np.zeros((H, W), dtype=float)
    nans = np.all(is_nan, axis=0)  # (H, W)
    active_mask = ~nans

    n_active = active_mask.sum()
    weights_uni[active_mask] = 1.0 / n_active

    flat_weights_uni = weights_uni.ravel()
    # (no need to renormalize; it already sums to 1)

    # ---------------------------
    # Sampling
    # ---------------------------
    flat_indices_exp = np.random.choice(flat_weights_exp.size, size=N, p=probs_exp)
    rows_exp, cols_exp = np.unravel_index(flat_indices_exp, (H, W))

    flat_indices_uni = np.random.choice(
        flat_weights_uni.size, size=N, p=flat_weights_uni
    )
    rows_uni, cols_uni = np.unravel_index(flat_indices_uni, (H, W))

    # ---------------------------
    # Plots (if requested)
    # ---------------------------
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Background: mean over normalized rasters
        bg = np.nanmean(tensor, axis=0)  # (H, W)

        # (a) overlay exp-based samples
        plt.figure(figsize=(6, 5))
        plt.title(f"Exp-based samples (N={N})")
        plt.imshow(bg, cmap="viridis")
        plt.scatter(cols_exp, rows_exp, s=5, c="red", marker="s")
        plt.tight_layout()
        out_path_exp = out_dir / f"{base_name}_exp.png"
        plt.savefig(out_path_exp, dpi=150)
        plt.close()
        print(f"[DEBUG] Saved exp-based samples overlay to {out_path_exp}")

        # (b) overlay uniform samples
        plt.figure(figsize=(6, 5))
        plt.title(f"Uniform samples (N={N})")
        plt.imshow(bg, cmap="viridis")
        plt.scatter(cols_uni, rows_uni, s=5, c="red", marker="s")
        plt.tight_layout()
        out_path_uni = out_dir / f"{base_name}_uniform.png"
        plt.savefig(out_path_uni, dpi=150)
        plt.close()
        print(f"[DEBUG] Saved uniform samples overlay to {out_path_uni}")

        # (c) heatmap of exp-based probability distribution
        probs_exp_2d = probs_exp.reshape(H, W)

        plt.figure(figsize=(6, 5))
        plt.title("Exp-based probability heatmap")
        plt.imshow(probs_exp_2d, cmap="viridis")
        plt.colorbar(label="p_exp(h, w)")
        plt.tight_layout()
        out_path_heat = out_dir / f"{base_name}_exp_heatmap.png"
        plt.savefig(out_path_heat, dpi=150)
        plt.close()
        print(f"[DEBUG] Saved exp-based probability heatmap to {out_path_heat}")

    return rows_exp, cols_exp, rows_uni, cols_uni


# ---------------------------------------------------------------------------
# Main simulation entry point
# ---------------------------------------------------------------------------


def run_simulation(N_samples: int = 1000):
    """
    1. Load & normalize rasters (filtered by GEO_FEATURES_DISTRIBUTION).
    2. Produce debug plots (per-raster image + unique histogram).
    3. Build a probability distribution over grid cells from the normalized stack.
    4. Sample N cells from:
         - exp-based distribution,
         - uniform distribution over the same support.
    5. Plot both overlays + probability heatmap.
    """
    print("Running simulation... (this is a placeholder function)")

    tifs = TIFF_OUT_DIR.glob("*.tif")
    tensor, names = stack_rasters(tifs)
    print("Stack shape (M, H, W):", tensor.shape)
    print("Channels:", names)

    debug_dir = TIFF_OUT_DIR / "simulation_plots"

    rows_exp, cols_exp, rows_uni, cols_uni = sample_cells(
        tensor,
        N=N_samples,
        grid=None,  # or pass a (H, W) mask if you want
        out_dir=debug_dir,
        base_name="samples_overlay",
    )

    print(f"Sampled {N_samples} cells (exp-based and uniform).")
