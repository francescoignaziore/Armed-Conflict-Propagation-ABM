import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.ndimage import binary_dilation, label
from geo_sim.config.paths import TIFF_OUT_DIR
import csv


PROB_MAP_START = TIFF_OUT_DIR / "simulation_plots" / "onset_weights_pop+viirs.tif"
PROB_MAP_MOVE = TIFF_OUT_DIR / "simulation_plots" / "onset_weights_pop+roads.tif"

DOWNSAMPLE_FACTOR = 5
N_GROUPS = 10
SEED = np.random.randint(0, 2**31 - 1)

M_SCALAR = 50.0
GAMMA = 0.30
GROWTH_POWER = 1.0
R_BASE = 0.00
R_BONUS = 0.30


SIGMA_BASE = 1
T_MAX = 5000
SAVE_INTERVAL = 100  # For saving frames
STATS_INTERVAL = 100  # For saving per-group stats
SAMPLING_POWER = 1.0
MAX_POSSIBLE_STRENGTH = M_SCALAR * 50.0
FRAGMENT_INTERVAL = 100  # check fragmentation every 100 steps

# Payoff matrix
PAYOFF_R = 0.05  # Reward: mutual peace
PAYOFF_T = 0.50  # Temptation: predation profitable
PAYOFF_P = -9.0  # Punishment: mutual defection ("meat grinder")
PAYOFF_S = -10.0  # Sucker: exploited, instant death-level

INTERACTION_RATE = 1.0
PD_GROWTH_SCALE = 0.01  # PD effect scaling when mapped into growth rate

# Thresholds
PRESENCE_EPSILON = 1
CLEAN_EPSILON = 1e-3


def load_and_downsample(path, factor):
    if not path.exists():
        return None
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
    if data.max() > 0:
        data /= data.max()
    return data


def compute_fragmentation(S, epsilon=CLEAN_EPSILON):
    """
    For each group g, count how many connected components it has.
    Returns an array of shape (N_GROUPS,) with component counts.
    """
    N_GROUPS, H, W = S.shape
    frag_counts = np.zeros(N_GROUPS, dtype=np.int32)

    # 4-connectivity (up, down, left, right). Use None for 8-connectivity.
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    for g in range(N_GROUPS):
        mask = S[g] > epsilon
        if not np.any(mask):
            frag_counts[g] = 0
            continue
        _, n_comp = label(mask, structure=structure)
        frag_counts[g] = n_comp

    return frag_counts


def apply_border_interaction(S, move_matrix, interaction_rate=1.0):
    """Legacy pixel-level border interaction (unused, kept for reference)."""
    N, H, W = S.shape
    delta_S = np.zeros_like(S)
    masks = [grp > PRESENCE_EPSILON for grp in S]

    for i in range(N):
        my_mask = masks[i]
        if not np.any(my_mask):
            continue

        for j in range(N):
            if i == j:
                continue
            enemy_mask = masks[j]
            if not np.any(enemy_mask):
                continue

            enemy_influence = binary_dilation(enemy_mask, iterations=1)
            front_line = my_mask & enemy_influence

            if not np.any(front_line):
                continue

            p_i = move_matrix[i, j]
            p_j = move_matrix[j, i]

            payoff = (
                (p_i * p_j * PAYOFF_R)
                + (p_i * (1 - p_j) * PAYOFF_S)
                + ((1 - p_i) * p_j * PAYOFF_T)
                + ((1 - p_i) * (1 - p_j) * PAYOFF_P)
            )
            delta_S[i][front_line] += S[i][front_line] * payoff * interaction_rate

    return delta_S


def resolve_strategies(N, types, memory):
    """Returns a matrix moves[i, j] = p(C | i vs j)."""
    current_moves = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            t = types[i]
            if t == "ALL_C":
                p = 1.0
            elif t == "ALL_D":
                p = 0.0
            elif t == "RAND":
                p = 0.5
            elif t == "TFT":
                p = memory[i, j]
            else:
                p = 0.5
            current_moves[i, j] = p
    return current_moves


def run_simulation():
    np.random.seed(SEED)
    print("Running Siege Model (Labeled)")
    print(f"Random seed: {SEED}")

    raw_start = load_and_downsample(PROB_MAP_START, DOWNSAMPLE_FACTOR)
    raw_move = load_and_downsample(PROB_MAP_MOVE, DOWNSAMPLE_FACTOR)
    if raw_start is None or raw_move is None:
        print("Missing probability maps. Aborting.")
        return

    # Capacity / movement fields
    K = M_SCALAR * np.power(raw_move, GAMMA)
    Mu = 0.05 + 0.95 * np.power(raw_move, GAMMA)

    # Background for plotting
    K_bg = np.log1p(K)
    K_bg_max = np.percentile(K_bg, 99)
    # Precompute indices for COM / radius
    rows_idx, cols_idx = np.indices(K.shape)

    # Precompute high-capacity mask (top 25% of K)
    cap_threshold = np.quantile(K, 0.75)
    high_cap_mask = K >= cap_threshold

    S = np.zeros((N_GROUPS, *K.shape), dtype=np.float32)

    flat_probs = np.power(raw_start.flatten(), SAMPLING_POWER)
    flat_probs /= flat_probs.sum()
    seed_indices = np.random.choice(
        flat_probs.size, size=N_GROUPS, p=flat_probs, replace=False
    )

    # Strategy types per group
    avail = ["ALL_D", "ALL_C", "TFT", "RAND"]
    probs = [0.25, 0.25, 0.25, 0.25]
    group_types = np.random.choice(avail, size=N_GROUPS, p=probs)
    memory = np.ones((N_GROUPS, N_GROUPS), dtype=np.float32)

    # Colors for plotting
    strat_colors = []
    for t in group_types:
        if t == "ALL_D":
            c = "#FF0000"  # Red
        elif t == "ALL_C":
            c = "#00FF00"  # Green
        elif t == "TFT":
            c = "#0000FF"  # Blue
        else:
            c = "#FFFF00"
        strat_colors.append(plt.matplotlib.colors.to_rgb(c))

    cmap_id = cm.get_cmap("nipy_spectral", N_GROUPS)
    id_colors = [cmap_id(i)[:3] for i in range(N_GROUPS)]

    print("\nSpawning groups:")
    for g in range(N_GROUPS):
        sx, sy = np.unravel_index(seed_indices[g], K.shape)
        S[g, sx, sy] = max(K[sx, sy], 100.0)
        print(f"  G{g} ({group_types[g]}): {sx},{sy}")

    frames_dir = (
        TIFF_OUT_DIR / "simulation_plots" / "group_based_loss" / f"frames_{SEED}"
    )
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Stats collection
    stats_rows = []
    prev_stats_strength = np.zeros(N_GROUPS, dtype=np.float32)

    print(f"\nSimulating {T_MAX} steps...")
    fragment_records = []  # list of (t, frag_g0, frag_g1, ..., frag_gN)

    for t in range(1, T_MAX + 1):
        # Strategic layer: global PD -> group-wide growth modifier (only when in contact)
        moves = resolve_strategies(N_GROUPS, group_types, memory)

        # Masks for presence of each group (significant presence)
        masks = [grp > PRESENCE_EPSILON for grp in S]

        # Dilated masks to detect adjacency (front lines)
        dilated_masks = []
        for m in masks:
            if np.any(m):
                dilated_masks.append(binary_dilation(m, iterations=1))
            else:
                dilated_masks.append(m)

        # Total strength per group (before physics)
        group_strength = np.array([grp.sum() for grp in S], dtype=np.float32)

        group_payoff = np.zeros(N_GROUPS, dtype=np.float32)
        group_payoff_sym = np.zeros(N_GROUPS, dtype=np.float32)
        group_payoff_asym = np.zeros(N_GROUPS, dtype=np.float32)

        for i in range(N_GROUPS):
            if not np.any(masks[i]):
                continue
            for j in range(N_GROUPS):
                if i == j:
                    continue
                if not np.any(masks[j]):
                    continue

                # Only count PD if there is a front line / adjacency
                front_line = masks[i] & dilated_masks[j]
                if not np.any(front_line):
                    continue

                p_i = moves[i, j]
                p_j = moves[j, i]

                # Symmetric part: (C,C) and (D,D) -> size-scaled
                sym_cc = p_i * p_j * PAYOFF_R
                sym_dd = (1 - p_i) * (1 - p_j) * PAYOFF_P
                payoff_sym = sym_cc + sym_dd

                # Asymmetric part: (C,D) and (D,C) -> NOT size-scaled
                asym_cd = p_i * (1 - p_j) * PAYOFF_S
                asym_dc = (1 - p_i) * p_j * PAYOFF_T
                payoff_asym = asym_cd + asym_dc

                # Relative size factor (only applied to symmetric part)
                size_i = group_strength[i]
                size_j = group_strength[j]
                denom = size_i + size_j + 1e-9
                relative_factor = size_j / denom

                contrib_sym = relative_factor * payoff_sym
                contrib_asym = payoff_asym

                group_payoff_sym[i] += contrib_sym
                group_payoff_asym[i] += contrib_asym
                group_payoff[i] += contrib_sym + contrib_asym

        # Physics: growth + excess + road-biased flow
        k_ratio = K / M_SCALAR
        growth = np.power(k_ratio, GROWTH_POWER)
        local_r = R_BASE + (R_BONUS * growth)

        S_tilde = np.empty_like(S)
        for g in range(N_GROUPS):
            effective_r = local_r + INTERACTION_RATE * PD_GROWTH_SCALE * group_payoff[g]
            # Optional safety clamp:
            # effective_r = np.maximum(effective_r, -0.9)
            S_tilde[g] = np.minimum(
                S[g] * (1.0 + effective_r),
                MAX_POSSIBLE_STRENGTH,
            )

        excess = np.maximum(0, S_tilde - K)
        retained = np.minimum(S_tilde, K)

        total_S = np.sum(S, axis=0)
        influx = np.zeros_like(S)

        # Blocking and road-biased diffusion of excess
        for g in range(N_GROUPS):
            g_excess = excess[g]
            others_S = total_S - S[g]
            others_present = others_S > CLEAN_EPSILON  # any non-negligible other mass

            eff_K = K.copy()
            eff_K[others_present] = 0.0  # Cells blocked by presence of others

            pad_eff_K = np.pad(eff_K, 1, constant_values=0)
            pad_Mu = np.pad(Mu, 1, constant_values=0)

            K_up, Mu_up = pad_eff_K[0:-2, 1:-1], pad_Mu[0:-2, 1:-1]
            K_down, Mu_down = pad_eff_K[2:, 1:-1], pad_Mu[2:, 1:-1]
            K_left, Mu_left = pad_eff_K[1:-1, 0:-2], pad_Mu[1:-1, 0:-2]
            K_right, Mu_right = pad_eff_K[1:-1, 2:], pad_Mu[1:-1, 2:]

            path_up = (K + K_up) / (2 * M_SCALAR)
            mu_up = SIGMA_BASE + (1 - SIGMA_BASE) * path_up
            w_up = K_up * mu_up

            path_down = (K + K_down) / (2 * M_SCALAR)
            mu_down = SIGMA_BASE + (1 - SIGMA_BASE) * path_down
            w_down = K_down * mu_down

            path_left = (K + K_left) / (2 * M_SCALAR)
            mu_left = SIGMA_BASE + (1 - SIGMA_BASE) * path_left
            w_left = K_left * mu_left

            path_right = (K + K_right) / (2 * M_SCALAR)
            mu_right = SIGMA_BASE + (1 - SIGMA_BASE) * path_right
            w_right = K_right * mu_right

            sum_w = w_up + w_down + w_left + w_right + 1e-9

            f_up = g_excess * (w_up / sum_w)
            f_down = g_excess * (w_down / sum_w)
            f_left = g_excess * (w_left / sum_w)
            f_right = g_excess * (w_right / sum_w)

            influx[g, :-1, :] += f_up[1:, :]
            influx[g, 1:, :] += f_down[:-1, :]
            influx[g, :, :-1] += f_left[:, 1:]
            influx[g, :, 1:] += f_right[:, :-1]

        S = retained + influx

        # Memory update (TFT)
        memory = moves.T

        # Cleanup
        S = np.nan_to_num(S, nan=0.0, posinf=MAX_POSSIBLE_STRENGTH)
        S = np.minimum(S, MAX_POSSIBLE_STRENGTH)
        S[S < CLEAN_EPSILON] = 0.0

        # Total strength per group after cleanup
        group_strength_post = S.reshape(N_GROUPS, -1).sum(axis=1)

        # Group extinction after t >= 100
        dead_groups = np.zeros(N_GROUPS, dtype=bool)
        if t >= 100:
            dead_groups = group_strength_post < 500.0
            if np.any(dead_groups):
                S[dead_groups, :, :] = 0.0
        # Fragmentation tracking
        if t % FRAGMENT_INTERVAL == 0:
            frag_counts = compute_fragmentation(S, epsilon=CLEAN_EPSILON)
            # A group is "fragmented" if it has > 1 connected component
            fragmented_groups = np.where(frag_counts > 1)[0]

            # Store stats
            fragment_records.append((t, *frag_counts.tolist()))

            # Optional: print a quick summary
            if fragmented_groups.size > 0:
                frags_str = ", ".join(
                    f"G{g}={frag_counts[g]}" for g in fragmented_groups
                )
                print(f"[T={t}] Fragmented groups: {frags_str}")
            else:
                print(f"[T={t}] No fragmented groups.")

        # Stats collection
        if t % STATS_INTERVAL == 0:
            for g in range(N_GROUPS):
                total = float(group_strength_post[g])
                alive = total > 0.0
                extinct = bool(dead_groups[g])

                # Area where group is significantly present
                area = int((S[g] > PRESENCE_EPSILON).sum())
                mean_strength = total / area if area > 0 else 0.0

                if total > 0.0:
                    mass = S[g]
                    cx = float((rows_idx * mass).sum() / total)
                    cy = float((cols_idx * mass).sum() / total)

                    dx = rows_idx - cx
                    dy = cols_idx - cy
                    rg = float(np.sqrt(((dx * dx + dy * dy) * mass).sum() / total))

                    mass_high = float(mass[high_cap_mask].sum())
                    share_high = mass_high / total if total > 0 else 0.0
                else:
                    cx = np.nan
                    cy = np.nan
                    rg = 0.0
                    share_high = 0.0

                # Neighbors and frontline cells
                num_adj = 0
                frontline_cells = 0
                coop_neigh_vals = []
                for j in range(N_GROUPS):
                    if j == g:
                        continue
                    if not np.any(masks[j]):
                        continue
                    front_line = masks[g] & dilated_masks[j]
                    if np.any(front_line):
                        num_adj += 1
                        frontline_cells += int(front_line.sum())
                        coop_neigh_vals.append(moves[g, j])

                avg_p = float(moves[g].sum() / (N_GROUPS - 1))
                avg_p_neigh = (
                    float(np.mean(coop_neigh_vals))
                    if len(coop_neigh_vals) > 0
                    else np.nan
                )

                delta_total = float(total - prev_stats_strength[g])

                row = {
                    "t": t,
                    "group_id": g,
                    "strategy_type": group_types[g],
                    "alive_flag": int(alive),
                    "extinct_flag": int(extinct),
                    "total_strength": total,
                    "delta_total_strength": delta_total,
                    "area_cells": area,
                    "mean_strength_per_cell": mean_strength,
                    "com_x": cx,
                    "com_y": cy,
                    "radius_gyration": rg,
                    "share_high_capacity": share_high,
                    "num_adjacent_enemies": int(num_adj),
                    "frontline_cells": int(frontline_cells),
                    "group_payoff_total": float(group_payoff[g]),
                    "payoff_sym": float(group_payoff_sym[g]),
                    "payoff_asym": float(group_payoff_asym[g]),
                    "avg_p_cooperate": avg_p,
                    "avg_p_cooperate_neighbors": avg_p_neigh,
                }
                stats_rows.append(row)
                prev_stats_strength[g] = total

        # Plotting
        if t % SAVE_INTERVAL == 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.axis("off")
            ax2.axis("off")

            ax1.imshow(K_bg, cmap="Greys", alpha=0.5, vmax=K_bg_max)
            ax2.imshow(K_bg, cmap="Greys", alpha=0.5, vmax=K_bg_max)

            total_mass_map = np.sum(S, axis=0)
            has_mass = total_mass_map > CLEAN_EPSILON

            if np.any(has_mass):
                dominant = np.argmax(S, axis=0)
                rgb_strat = np.zeros((*K.shape, 4))
                rgb_id = np.zeros((*K.shape, 4))

                for g in range(N_GROUPS):
                    mask = (dominant == g) & has_mass
                    if not np.any(mask):
                        continue

                    # Strategy color (left)
                    rgb_strat[mask, :3] = strat_colors[g]
                    rgb_strat[mask, 3] = 0.8

                    # Identity color (right)
                    rgb_id[mask, :3] = id_colors[g]
                    rgb_id[mask, 3] = 0.8

                    # Labels: total strength per group
                    rows, cols = np.where(mask)
                    cy_lab, cx_lab = np.mean(rows), np.mean(cols)

                    current_strength = float(S[g].sum())
                    label_text = f"{int(current_strength)}"

                    for ax in (ax1, ax2):
                        ax.text(
                            cx_lab,
                            cy_lab,
                            label_text,
                            color="black",
                            fontsize=9,
                            weight="bold",
                            ha="center",
                            va="center",
                            bbox=dict(
                                facecolor="white",
                                alpha=0.7,
                                edgecolor="none",
                                pad=1.5,
                            ),
                        )

                ax1.imshow(rgb_strat)
                ax2.imshow(rgb_id)

            ax1.set_title(f"Strategy (Politics)\nRed=War, Green=Peace", fontsize=14)
            ax2.set_title(f"Identity (Groups)", fontsize=14)
            fig.suptitle(f"Step T={t:03d}", fontsize=16)

            out_path = frames_dir / f"frame_{t:03d}.png"
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print(f"   Step {t} -> Saved")

    # Write stats CSV
    stats_path = frames_dir / f"stats_{SEED}.csv"
    if stats_rows:
        fieldnames = [
            "t",
            "group_id",
            "strategy_type",
            "alive_flag",
            "extinct_flag",
            "total_strength",
            "delta_total_strength",
            "area_cells",
            "mean_strength_per_cell",
            "com_x",
            "com_y",
            "radius_gyration",
            "share_high_capacity",
            "num_adjacent_enemies",
            "frontline_cells",
            "group_payoff_total",
            "payoff_sym",
            "payoff_asym",
            "avg_p_cooperate",
            "avg_p_cooperate_neighbors",
        ]
        with open(stats_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats_rows)
        print(f"Saved stats to {stats_path}")
    else:
        print("No stats rows collected.")
    # Save fragmentation stats
    if fragment_records:
        csv_path = frames_dir / "fragmentation_stats.csv"
        header = ["t"] + [f"group_{g}_components" for g in range(N_GROUPS)]

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(fragment_records)

        print(f"Saved fragmentation stats to {csv_path}")

    print("Done.")


if __name__ == "__main__":
    run_simulation()
