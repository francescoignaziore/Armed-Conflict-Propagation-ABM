import pdb
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import numpy as np
import rasterio
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

import matplotlib

matplotlib.use("Agg")  # HPC / headless
import matplotlib.pyplot as plt

from geo_sim.config.paths import TIFF_OUT_DIR
from geo_sim.config.consts import (
    SIMULATION_COMBINATIONS,
    GroupInitStrategy
)

# ---------------------------------------------------------------------------
# Helpers: normalization
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
        return np.full_like(arr, np.nan, dtype=float)

    vmin = float(vals.min())
    vmax = float(vals.max())
    out = arr.copy()

    if vmax > vmin:
        out[~mask_invalid] = (out[~mask_invalid] - vmin) / (vmax - vmin)
    else:
        out[~mask_invalid] = 0.0

    out[mask_invalid] = np.nan
    return out


# ---------------------------------------------------------------------------
# Data Loading (The Bank)
# ---------------------------------------------------------------------------


def load_feature_bank(
    paths: Iterable[Path],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Loads all TIFFs into a dictionary {name: normalized_array}.
    Enforces the VIIRS mask as the 'Territory' (Master Mask).
    """
    bank: Dict[str, np.ndarray] = {}
    path_list = list(paths)

    # 1. Establish Master Mask from VIIRS
    # We assume VIIRS defines the valid country border (NaN outside).
    viirs_path = next((p for p in path_list if "viirs" in p.stem.lower()), None)
    if not viirs_path:
        raise ValueError("VIIRS file is required to define the Master Mask (Border).")

    with rasterio.open(viirs_path) as ds:
        viirs_raw = ds.read(1)
        master_mask = np.isfinite(viirs_raw)
        if ds.nodata is not None:
            master_mask &= viirs_raw != ds.nodata

    print(f"[Loader] Master Mask established. Valid cells: {master_mask.sum()}")

    # 2. Load and Transform Layers
    # The knob for population extremeness
    POP_SENSITIVITY = 1.0

    for p in path_list:
        name = None
        # Heuristic to identify feature from filename
        if "viirs" in p.stem.lower():
            name = "viirs"
        elif "pop" in p.stem.lower():
            name = "pop"
        elif "roads" in p.stem.lower():
            name = "roads"
        elif "water" in p.stem.lower():
            name = "water"
        elif "natural" in p.stem.lower():
            name = "natural"
        elif "landuse" in p.stem.lower():
            name = "landuse"

        if name is None:
            continue

        with rasterio.open(p) as ds:
            arr = ds.read(1).astype(float)
            nodata = ds.nodata

        # Apply Master Mask: everything outside border becomes NaN
        arr[~master_mask] = np.nan

        # Handle Internal Nodata (inside country but no data) -> 0.0
        inside_but_invalid = master_mask & (np.isnan(arr) | (arr == nodata))
        arr[inside_but_invalid] = 0.0

        # --- FEATURE SPECIFIC TRANSFORMS ---
        valid_mask = np.isfinite(arr)

        if name == "pop":
            # Log transform -> Normalize -> Exp Transform
            arr[valid_mask] = np.log1p(arr[valid_mask])
            arr = _normalize_01_with_nodata(arr, np.nan)
            arr[valid_mask] = np.exp(arr[valid_mask] * POP_SENSITIVITY)
            # Renormalize back to 0-1 for fair summation
            arr = _normalize_01_with_nodata(arr, np.nan)

        elif name == "viirs":
            arr[valid_mask] = np.log1p(arr[valid_mask])
            arr = _normalize_01_with_nodata(arr, np.nan)

        else:
            arr = _normalize_01_with_nodata(arr, np.nan)

        bank[name] = arr
        print(f"[Loader] Loaded feature: {name}")

    return bank, master_mask


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_strategy(
    feature_keys: List[str],
    bank: Dict[str, np.ndarray],
    master_mask: np.ndarray,
    N: int,
    intensity_factor: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sums normalized layers, applies exponential weighting, samples N points.
    """
    H, W = master_mask.shape
    combined_score = np.zeros((H, W), dtype=float)

    for key in feature_keys:
        if key not in bank:
            print(f"[Warning] Key '{key}' not found in feature bank. Skipping.")
            continue
        layer = bank[key]
        valid_vals = np.where(np.isfinite(layer), layer, 0.0)
        combined_score += valid_vals

    if combined_score.sum() == 0:
        return np.array([]), np.array([])

    # Apply Physics (Intensity)
    score = combined_score * intensity_factor
    score = np.clip(score, -100, 100)
    weights = np.exp(score)

    # Apply Mask & Hard Zeros
    weights[~master_mask] = 0.0
    weights[combined_score <= 1e-9] = 0.0

    flat_weights = weights.ravel()
    total_w = flat_weights.sum()

    if total_w == 0:
        raise ValueError(f"Strategy {feature_keys} has zero total weight.")

    probs = flat_weights / total_w
    flat_indices = np.random.choice(flat_weights.size, size=N, p=probs)
    rows, cols = np.unravel_index(flat_indices, (H, W))

    return rows, cols


def sample_uniform(master_mask: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    H, W = master_mask.shape
    weights = np.zeros((H, W), dtype=float)
    weights[master_mask] = 1.0

    flat_weights = weights.ravel()
    probs = flat_weights / flat_weights.sum()
    flat_indices = np.random.choice(flat_weights.size, size=N, p=probs)
    rows, cols = np.unravel_index(flat_indices, (H, W))
    return rows, cols


# ---------------------------------------------------------------------------
# Comparative Plotting
# ---------------------------------------------------------------------------


def plot_nnd_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    max_samples_for_calc: int = 5000,
) -> None:
    """
    Generates TWO plots:
    1. Discrete (Frequency Polygon / Line Plot) - requested "lines as before"
    2. Continuous (Log-Gaussian KDE)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # --- PRE-CALCULATE NNDs ---
    nnd_data = {}
    for label, (rows, cols) in results.items():
        if len(rows) == 0:
            continue
        if len(rows) > max_samples_for_calc:
            idx_sub = np.random.choice(len(rows), max_samples_for_calc, replace=False)
            pts = np.column_stack((rows[idx_sub], cols[idx_sub]))
        else:
            pts = np.column_stack((rows, cols))

        if len(pts) < 2:
            continue
        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)
        nnd_data[label] = dists[:, 1]

    # ==========================================
    # PLOT 1: DISCRETE (Frequency Polygon)
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Bins aligned to integers
    bins = np.linspace(0, 40, 41)

    for idx, (label, nnd) in enumerate(nnd_data.items()):
        if label.lower() == "uniform":
            c, ls, lw, alpha, zorder = "black", "--", 2, 0.7, 0
        else:
            c, ls, lw, alpha, zorder = colors[idx], "-", 2, 0.7, 10

        # Compute histogram explicitly
        counts, bin_edges = np.histogram(nnd, bins=bins, density=True)
        # Calculate centers
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot lines connecting centers
        plt.plot(
            centers,
            counts,
            label=label,
            color=c,
            linestyle=ls,
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
            marker="o",  # Optional: markers help show the discrete steps
            markersize=3,
        )

    plt.title("Spatial Clustering: NND (Discrete Line Plot)")
    plt.xlabel("Distance to Nearest Neighbor (Grid Cells)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 40)
    plt.ylim(0, 0.30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_all_NND_discrete.png", dpi=150)
    plt.close()

    # ==========================================
    # PLOT 2: CONTINUOUS (Log-Gaussian KDE)
    # ==========================================
    plt.figure(figsize=(10, 6))
    x_grid = np.linspace(0.1, 40, 200)

    for idx, (label, nnd) in enumerate(nnd_data.items()):
        nnd_pos = nnd[nnd > 0]
        if len(nnd_pos) < 2:
            continue

        try:
            kde = gaussian_kde(np.log(nnd_pos))
            log_pdf = kde(np.log(x_grid))
            y_vals = log_pdf / x_grid

            if label.lower() == "uniform":
                c, ls, lw, alpha, zorder = "black", "--", 2, 0.7, 0
            else:
                c, ls, lw, alpha, zorder = colors[idx], "-", 2, 0.7, 10

            plt.plot(
                x_grid,
                y_vals,
                label=label,
                color=c,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )

        except np.linalg.LinAlgError:
            continue

    plt.title("Spatial Clustering: NND (Log-Gaussian KDE)")
    plt.xlabel("Distance to Nearest Neighbor (Grid Cells)")
    plt.ylabel("Probability Density")
    plt.xlim(0, 40)
    plt.ylim(0, 0.30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_all_NND_continuous.png", dpi=150)
    plt.close()


def plot_multi_strategy_exposure(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    bank: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """
    For *each* feature in the bank, plot a histogram showing how *each* strategy
    is exposed to it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for feature_name, feature_arr in bank.items():
        plt.figure(figsize=(10, 6))

        for idx, (strat_name, (rows, cols)) in enumerate(results.items()):
            if len(rows) == 0:
                continue

            vals = feature_arr[rows, cols]
            vals = vals[np.isfinite(vals)]

            if strat_name.lower() == "uniform":
                c, ls, alpha, fill = "black", "--", 1.0, False
            else:
                c, ls, alpha, fill = colors[idx], "-", 0.5, True

            plt.hist(
                vals,
                bins=40,
                range=(0, 1),
                density=True,
                histtype="step" if not fill else "stepfilled",
                label=strat_name,
                color=c,
                linestyle=ls,
                alpha=alpha if fill else 1.0,
                linewidth=1.5 if not fill else 0,
            )
            if fill:
                plt.hist(
                    vals,
                    bins=40,
                    range=(0, 1),
                    density=True,
                    histtype="step",
                    color=c,
                    linewidth=1.0,
                )

        plt.title(f"Cross-Exposure: Exposure to {feature_name}")
        plt.xlabel(f"Normalized {feature_name} Intensity (0-1)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"exposure_to_{feature_name}.png", dpi=150)
        plt.close()


def plot_multi_scale_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    radii_km: List[float] = [1, 2, 5, 10, 20, 50, 100],
) -> None:
    """
    One line plot comparing average neighbor counts vs radius.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    print("[Analysis] Calculating Multi-Scale Density...")

    for idx, (label, (rows, cols)) in enumerate(results.items()):
        points = np.column_stack((rows, cols))
        if len(points) == 0:
            continue

        tree = cKDTree(points)
        avg_counts = []

        for r in radii_km:
            counts = tree.query_ball_point(points, r=r, return_length=True)
            avg = np.mean(np.array(counts) - 1)
            avg_counts.append(avg)

        if label.lower() == "uniform":
            plt.plot(
                radii_km,
                avg_counts,
                label=label,
                color="black",
                linestyle="--",
                marker="x",
                linewidth=2,
            )
        else:
            plt.plot(
                radii_km,
                avg_counts,
                label=label,
                color=colors[idx],
                marker="o",
                alpha=0.8,
            )

    plt.title("Multi-Scale Density Analysis")
    plt.xlabel("Radius (grid cells/km)")
    plt.ylabel("Avg Neighbors")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_multiscale.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Advanced Analytics (New)
# ---------------------------------------------------------------------------


def plot_spatial_lorenz(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    master_mask: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Plots Lorenz Curves to visualize Spatial Inequality (Concentration).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))

    H, W = master_mask.shape
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Perfect Uniformity Line
    plt.plot([0, 1], [0, 1], color="black", linestyle=":", label="Perfect Uniformity")

    for idx, (label, (rows, cols)) in enumerate(results.items()):
        if len(rows) == 0:
            continue

        density_grid = np.zeros((H, W), dtype=float)
        np.add.at(density_grid, (rows, cols), 1)
        valid_vals = density_grid[master_mask]
        sorted_vals = np.sort(valid_vals)

        cum_agents = np.cumsum(sorted_vals) / sorted_vals.sum()
        cum_area = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        auc = np.trapz(cum_agents, cum_area)
        gini = 1.0 - 2.0 * auc

        if label.lower() == "uniform":
            c, ls = "black", "--"
        else:
            c, ls = colors[idx], "-"

        plt.plot(
            cum_area,
            cum_agents,
            label=f"{label} (Gini={gini:.2f})",
            color=c,
            linestyle=ls,
        )

    plt.title("Spatial Lorenz Curve (Concentration)")
    plt.xlabel("Cumulative Fraction of Territory Area")
    plt.ylabel("Cumulative Fraction of Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_lorenz_curve.png", dpi=150)
    plt.close()


def plot_mst_cost(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    max_nodes: int = 1000,
) -> None:
    """
    Calculates the Total Length of the Minimum Spanning Tree (MST).
    """
    costs = []
    labels = []

    print("[Analysis] Calculating MST Costs (Subsampled)...")

    for label, (rows, cols) in results.items():
        if len(rows) == 0:
            continue

        if len(rows) > max_nodes:
            idx = np.random.choice(len(rows), max_nodes, replace=False)
            pts = np.column_stack((rows[idx], cols[idx]))
        else:
            pts = np.column_stack((rows, cols))

        if len(pts) < 2:
            costs.append(0)
            labels.append(label)
            continue

        dist_matrix = squareform(pdist(pts))
        mst = minimum_spanning_tree(dist_matrix)
        total_len = mst.toarray().sum().sum() / 2.0
        costs.append(total_len)
        labels.append(label)

    plt.figure(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(costs)))
    bars = plt.bar(labels, costs, color=colors, alpha=0.7, edgecolor="black")

    for i, label in enumerate(labels):
        if label.lower() == "uniform":
            bars[i].set_color("gray")
            bars[i].set_hatch("//")

    plt.title(f"Infrastructure Cost Proxy: MST Length (N={max_nodes})")
    plt.ylabel("Total Grid Distance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_mst_cost.png", dpi=150)
    plt.close()


def plot_void_distribution(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    master_mask: np.ndarray,
    out_dir: Path,
    n_probes: int = 5000,
) -> None:
    """
    Plots distribution of distance from RANDOM points to the NEAREST AGENT.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))

    h_idx, w_idx = np.where(master_mask)
    if len(h_idx) > n_probes:
        idx = np.random.choice(len(h_idx), n_probes, replace=False)
        probe_pts = np.column_stack((h_idx[idx], w_idx[idx]))
    else:
        probe_pts = np.column_stack((h_idx, w_idx))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    x_grid = np.linspace(0, 100, 200)

    for idx, (label, (rows, cols)) in enumerate(results.items()):
        agent_pts = np.column_stack((rows, cols))
        if len(agent_pts) == 0:
            continue

        tree = cKDTree(agent_pts)
        dists, _ = tree.query(probe_pts, k=1)

        try:
            kde = gaussian_kde(dists)
            y_vals = kde(x_grid)

            if label.lower() == "uniform":
                c, ls = "black", "--"
            else:
                c, ls = colors[idx], "-"

            plt.plot(x_grid, y_vals, label=label, color=c, linestyle=ls)
        except:
            pass

    plt.title("Void Analysis: Distance from Random Point to Nearest Agent")
    plt.xlabel("Distance (Grid Cells)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "comparison_void_dist.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------------

def init_grid(G: int, sampling_strategy=GroupInitStrategy.UNIFORM):
    """
    Initialize groups on the grid
    The grid is currently loaded from a fixed path

    Returns: H, W, rows, cols
    - H,W: shape of the grid
    - rows/cols: the x/y indices of the groups
    """
    # 1. Load Data
    tifs = TIFF_OUT_DIR.glob("*.tif")
    bank, master_mask = load_feature_bank(tifs)
    H,W = master_mask.shape
    if not bank:
        print("No features found! Exiting.")
        return

    # 2. Sample groups
    N_samples = G
    rows, cols = None
    if sampling_strategy==GroupInitStrategy.UNIFORM:
        ## 2a. Uniform
        rows, cols = sample_uniform(master_mask, N_samples)
    elif sampling_strategy==GroupInitStrategy.POP:
        combo = ['pop']
        rows, cols = sample_strategy(combo, bank, master_mask, N_samples)
    elif sampling_strategy==GroupInitStrategy.POP_VIIRS:
        combo = ["pop", "viirs"]
        rows, cols = sample_strategy(combo, bank, master_mask, N_samples)
    elif sampling_strategy==GroupInitStrategy.POP_ROADS:
        combo = ["pop", "roads"]
        rows, cols = sample_strategy(combo, bank, master_mask, N_samples)
    elif sampling_strategy==GroupInitStrategy.POP_VIIRS_ROADS:
        combo = ["pop", "viirs", "roads"]
        rows, cols = sample_strategy(combo, bank, master_mask, N_samples)
    elif sampling_strategy==GroupInitStrategy.ROADS:
        combo = ["roads"]
        rows, cols = sample_strategy(combo, bank, master_mask, N_samples)
    else:
        raise NotImplementedError()
    
    return H,W,rows,cols

def run_simulation(N_samples: int = 10000):
    """
    1. Load Bank (Masked by VIIRS).
    2. Run Uniform Sampler.
    3. Run Samplers for every combination in SIMULATION_COMBINATIONS.
    4. Generate Comparative Plots.
    """
    print(f"Running Multi-Strategy Simulation with N={N_samples}...")
    debug_dir = TIFF_OUT_DIR / "simulation_plots"

    # 1. Load Data
    tifs = TIFF_OUT_DIR.glob("*.tif")
    bank, master_mask = load_feature_bank(tifs)
    if not bank:
        print("No features found! Exiting.")
        return

    results = {}

    # 2. Uniform Baseline
    print("Sampling Strategy: Uniform")
    u_rows, u_cols = sample_uniform(master_mask, N_samples)
    results["Uniform"] = (u_rows, u_cols)

    # 3. Iterate Strategies
    for combo in SIMULATION_COMBINATIONS:
        strat_name = "+".join(combo)
        print(f"Sampling Strategy: {strat_name}")

        rows, cols = sample_strategy(combo, bank, master_mask, N_samples)
        results[strat_name] = (rows, cols)

        # Quick Map
        bg = np.zeros_like(master_mask, dtype=float)
        if combo[0] in bank:
            bg = bank[combo[0]].copy()
            bg[~master_mask] = np.nan

        plt.figure(figsize=(6, 5))
        plt.imshow(bg, cmap="viridis", vmin=0, vmax=1)
        plt.scatter(cols, rows, s=1, c="red", alpha=0.5)
        plt.title(f"Sample Map: {strat_name}")
        plt.axis("off")
        plt.savefig(debug_dir / f"map_{strat_name}.png", dpi=100)
        plt.close()

    # 4. Compare
    print("Generating comparative plots...")
    plot_nnd_comparison(results, debug_dir)
    plot_multi_scale_comparison(results, debug_dir)
    plot_multi_strategy_exposure(results, bank, debug_dir)

    # Advanced
    print("Generating advanced analytics...")
    plot_spatial_lorenz(results, master_mask, debug_dir)
    plot_mst_cost(results, debug_dir)
    plot_void_distribution(results, master_mask, debug_dir)

    print(f"✓ Simulation complete. Check {debug_dir}")
