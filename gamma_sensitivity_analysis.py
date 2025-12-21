import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

from geo_sim.config.paths import TIFF_OUT_DIR

# Config (should match simulation settings)
PROB_MAP_MOVE = TIFF_OUT_DIR / "simulation_plots" / "onset_weights_pop+roads.tif"
DOWNSAMPLE_FACTOR = 5  # Same as your sim to keep scale consistent
M_SCALAR = 50.0  # Same as your sim

# Gammas to visualize (including your current 0.30)
GAMMA_VALUES = [0.15, 0.30, 1.0, 3.0]


def load_and_downsample(path, factor):
    if not path.exists():
        print(f"File not found: {path}")
        return None

    print(f"Loading {path.name}...")
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)

    if factor > 1:
        h, w = data.shape
        h_new, w_new = (h // factor) * factor, (w // factor) * factor
        data = (
            data[:h_new, :w_new]
            .reshape(h_new // factor, factor, w_new // factor, factor)
            .mean(axis=(1, 3))
        )

    # Normalize exactly as you do in the sim
    if data.max() > 0:
        data /= data.max()

    return data


def plot_gamma_sensitivity():
    raw_move = load_and_downsample(PROB_MAP_MOVE, DOWNSAMPLE_FACTOR)
    if raw_move is None:
        print("Could not load data. Check the path in the script.")
        return

    num_vars = len(GAMMA_VALUES)
    fig, axes = plt.subplots(2, num_vars, figsize=(5 * num_vars, 10))
    plt.suptitle(
        f"Sensitivity Analysis of Gamma on Real Data\nFile: {PROB_MAP_MOVE.name}",
        fontsize=16,
        y=0.98,
    )

    for i, gamma in enumerate(GAMMA_VALUES):
        K = M_SCALAR * np.power(raw_move, gamma)

        ax_map = axes[0, i]
        im = ax_map.imshow(K, cmap="inferno", vmin=0, vmax=M_SCALAR)

        label_type = "Base"
        if gamma < 1:
            label_type = "Smoothing" if gamma == 0.3 else "Smoothing"
        if gamma > 1:
            label_type = "Sharpening"

        ax_map.set_title(f"Gamma = {gamma}\n({label_type})", fontsize=12)
        ax_map.axis("off")

        if i == num_vars - 1:
            cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
            cbar.set_label("Capacity K")

        ax_hist = axes[1, i]
        flat_data = K.flatten()

        mask_nonzero = flat_data > 0.001
        data_to_plot = flat_data[mask_nonzero] if np.any(mask_nonzero) else flat_data

        ax_hist.hist(
            data_to_plot,
            bins=40,
            range=(0, M_SCALAR),
            color="#4c72b0",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_hist.set_xlabel("Capacity Value (K)")
        if i == 0:
            ax_hist.set_ylabel("Frequency (Log Scale)")

        ax_hist.set_yscale("log")
        ax_hist.grid(True, alpha=0.2, which="both")

        mean_val = np.mean(flat_data)
        p90 = np.percentile(flat_data, 90)
        ax_hist.annotate(
            f"Mean: {mean_val:.1f}\nTop 10%: >{p90:.1f}",
            xy=(0.05, 0.85),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        )

    plt.tight_layout()
    out_file = "gamma_sensitivity_real_data.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nPlot saved to {out_file}")
    plt.show()


if __name__ == "__main__":
    plot_gamma_sensitivity()
