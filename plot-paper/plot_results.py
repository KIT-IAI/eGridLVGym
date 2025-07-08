"""
Generate figures (Fig. 03, 04a–d, 05, 06, 08) from raw CSV logs of RL experiments.

Expected folder structure:
results/
├── household-economic/
│   ├── train_env_logs/fiveruns/     # monitor_*.csv
│   └── evaluations/                 # eval_*.csv, weights_scale/
├── feeder-voltage/                  # eval_*.csv
├── feeder-economic/                 # eval_*.csv, vde-shielding/
└── feeder-combined/                 # eval_*.csv

Usage:
    python plot_results.py

Output:
    Saves all figures as PNG and PGF in the current directory.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker as mtick
from matplotlib.patches import Patch
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

try:
    matplotlib.use("pgf")
except Exception:
    pass

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        "pgf.rcfonts": False,
        "font.size": 10,
    }
)

COLORS = sns.color_palette("muted", 15)
TRAINING_COLORS = {
    "ddpg": COLORS[6],
    "td3": COLORS[1],
    "sac": COLORS[0],
    "tqc": COLORS[2],
    "a2c": COLORS[8],
    "trpo": COLORS[9],
    "ppo": COLORS[7],
    "recurrent_ppo": COLORS[13],
    "capql": COLORS[4],
    "baseline_price_window": COLORS[5],
}

BASE_PATH = Path("../results")
SAVE_DIR = Path(".")
SAVE_DIR.mkdir(exist_ok=True)

# Helper


def _darken(rgb, factor: float = 0.7):
    hsv = rgb_to_hsv(rgb)
    hsv[2] *= factor
    return hsv_to_rgb(hsv)


#  Figure 03


def _load_training_curves(path: Path, algs: list[str]):
    df = pd.DataFrame()
    seeds: dict[str, list[str]] = {}
    for alg in algs:
        for fn in sorted(path.glob(f"4_{alg}_*.monitor.csv")):
            m = re.search(rf"4_{alg}_(\d+)\.monitor\.csv", fn.name)
            if not m:
                continue
            seed = m.group(1)
            col = f"{alg}_{seed}"
            cur = pd.read_csv(fn, skiprows=[0])
            df[col] = cur["r"].rolling(window=200).mean()
            seeds.setdefault(alg, []).append(seed)
    return df, seeds


def figure03_training_curves():
    sns.set_style("whitegrid", {"axes.grid": True, "axes.edgecolor": "black"})
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 10,
        }
    )
    style.use("tableau-colorblind10")

    algs = ["ddpg", "td3", "tqc", "sac", "a2c", "trpo", "ppo", "recurrent_ppo"]
    labels = ["DDPG", "TD3", "TQC", "SAC", "A2C", "TRPO", "PPO", "Recurrent PPO"]

    path = (
        BASE_PATH
        / "household-economic"
        / "train_env_logs"
        / "train_parallel_random_seeds_5"
    )
    scaling, total_eps, offset = 672, 8000, 150 * 672
    data, seeds = _load_training_curves(path, algs)

    fig, ax = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    for i, alg in enumerate(algs):
        cols = [f"{alg}_{s}" for s in seeds.get(alg, [])]
        if not cols:
            continue
        mean = data[cols].mean(axis=1)
        lo, hi = data[cols].min(axis=1), data[cols].max(axis=1)
        col = TRAINING_COLORS[alg]
        for sub in ax:
            x = mean.index * scaling
            sub.plot(x, mean, lw=1.5, color=col, label=labels[i])
            sub.fill_between(x, lo, hi, color=col, alpha=0.5, rasterized=True)

    for a in ax:
        a.set_xlim(offset, scaling * total_eps)
        a.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{int(x/10)}"))
    ax[1].set_ylim(200, 620.1)
    ax[0].set_ylabel("Score")
    ax[1].set_ylabel("Score")
    ax[1].set_xlabel("Environment Steps")
    ax[1].text(
        -0.09,
        -0.06,
        r"$\times 10^{1}$",
        transform=ax[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        clip_on=False,
    )
    fig.subplots_adjust(left=0.18)
    plt.tight_layout(rect=[0, 0, 1, 1])

    handles, l = ax[0].get_legend_handles_labels()
    for hh in handles:
        hh.set_linewidth(4.0)
    ax[0].legend(
        handles,
        l,
        loc="lower left",
        bbox_to_anchor=(-0.01, -0.02),
        ncol=2,
        frameon=True,
        handletextpad=0.3,
        columnspacing=0.5,
    )
    for hh in handles:
        hh.set_linewidth(1.0)
    fig.align_ylabels(ax[:])
    for ext in ("_rasterized.pgf", ".png"):
        plt.savefig(
            SAVE_DIR / f"fig03_training_performances_sorl{ext}", bbox_inches="tight"
        )
    plt.close(fig)


#  Figure 04a


def figure04a_bar() -> None:
    """Generate Fig. 04a – normalized scores including CAPQL and RBC itself.
    RBC appears as a grey bar fixed at 1.0 so readers can visually see the
    baseline.
    """
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 20,
        }
    )

    # Algorithms
    algs = [
        "ddpg",
        "td3",
        "tqc",
        "sac",
        "a2c",
        "trpo",
        "capql",
        "baseline_price_window",
    ]
    labels = ["DDPG", "TD3", "TQC", "SAC", "A2C", "TRPO", "CAPQL", "RBC"]

    # Colour palette
    pal = sns.color_palette("muted", 13)
    cmap = {
        "ddpg": pal[6],
        "td3": pal[1],
        "tqc": pal[2],
        "sac": pal[0],
        "a2c": pal[8],
        "trpo": pal[9],
        "capql": pal[4],
        "baseline_price_window": pal[5],
    }

    # Load CSVs
    scores: dict[str, float] = {}
    for alg in algs:
        fn = BASE_PATH / "household-economic" / "evaluations" / f"eval_{alg}.csv"
        if not fn.exists():
            print(f"[WARN] Missing {fn.name}")
            continue
        df = pd.read_csv(fn)
        if "cumulative_reward" not in df.columns:
            print(f"[WARN] cumulative_reward missing in {fn.name}")
            continue
        days = [df["cumulative_reward"].iloc[(d * 96 * 7) - 1] for d in range(1, 12)]
        if alg == "baseline_price_window":
            days = [-v for v in days]
        scores[alg] = np.mean(days)

    rbc_val = scores.get("baseline_price_window")
    if rbc_val is None or rbc_val == 0:
        raise ValueError("Reference value (RBC) not found or zero.")

    # Normalize
    norm = {
        alg: (1.0 if alg == "baseline_price_window" else scores[alg] / rbc_val)
        for alg in algs
        if alg in scores
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(False)
    ordered = [a for a in algs if a in norm]
    ax.set_xticks(range(len(ordered)))
    bars = ax.bar(
        range(len(ordered)),
        [norm[a] for a in ordered],
        color=[cmap[a] for a in ordered],
        width=0.7,
    )
    ax.set_xticklabels([labels[algs.index(a)] for a in ordered])

    for b, a in zip(bars, ordered):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{norm[a]:.2f}",
            ha="center",
            va="bottom",
            fontsize=18,
        )

    ax.set_ylabel("Normalized Score (RBC = 1.0)")
    ax.set_ylim(0, max(norm.values()) * 1.15)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    plt.tight_layout()

    for ext in ("png", "pgf"):
        plt.savefig(
            SAVE_DIR / f"fig04a_comparisons_CAPQL_SORL_algorithms_normalized.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


# Figure 04b


def figure04b_pareto():
    df = pd.read_csv(
        BASE_PATH
        / "household-economic"
        / "evaluations"
        / "weights_scale"
        / "eval_best_capql_combined.csv"
    )
    cost, flex = df["reward_cost"], df["reward_flexibility"]
    pts = np.c_[cost, flex]
    eff = np.ones(len(pts), bool)
    for i, p in enumerate(pts):
        dominated = (
            (pts[:, 0] >= p[0])
            & (pts[:, 1] >= p[1])
            & ((pts[:, 0] > p[0]) | (pts[:, 1] > p[1]))
        )
        if dominated.any():
            eff[i] = False
    pareto = df[eff].sort_values("reward_cost")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(cost, flex, alpha=0.4, color="gray", label="All solutions")
    ax.plot(
        pareto["reward_cost"],
        pareto["reward_flexibility"],
        color="red",
        lw=2,
        label="Pareto frontier",
    )
    ax.scatter(
        pareto["reward_cost"],
        pareto["reward_flexibility"],
        color="red",
        edgecolors="black",
        s=80,
        zorder=5,
    )
    ax.set_xlabel("Cost Reward")
    ax.set_ylabel("Flexibility Reward")
    ax.set_ylim(-15.0, 70.0)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    for ext in ("png", "pgf"):
        plt.savefig(SAVE_DIR / f"fig04b_pareto.{ext}", bbox_inches="tight")
    plt.close(fig)


#  Figure 04c and 04d


def figure04c_04d_3d():
    df = pd.read_csv(
        BASE_PATH
        / "household-economic"
        / "evaluations"
        / "weights_scale"
        / "eval_best_capql_linear.csv",
        index_col=0,
    )
    x = df["weight_cost"]
    y = df["weight_flexibility"]
    z_cost = df["reward_cost"]
    z_flex = df["reward_flexibility"]

    z_min, z_max = min(z_cost.min(), z_flex.min()), max(z_cost.max(), z_flex.max())
    zz_min, zz_max = int((z_min // 10) * 10), int((z_max // 10 + 1) * 10)
    z_ticks = np.arange(zz_min, zz_max + 1, 20)
    cmap = plt.cm.viridis
    plt.rcParams.update(
        {
            "font.size": 15,
        }
    )

    def _surf(z, label, stem):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(y, x, z, cmap=cmap, edgecolor="none", alpha=1.0)
        ax.set_xlabel("Flexibility Weight", fontsize=17, labelpad=5)
        ax.set_ylabel("Economic Weight", fontsize=17, labelpad=5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.20, 1.0, 5))
        ax.set_yticks(np.linspace(0.20, 1.0, 5))
        ax.set_zlim(z_min, z_max)
        ax.set_zticks(z_ticks)
        ax.invert_xaxis()
        ax.view_init(elev=20, azim=-50)
        ax.dist = 12
        norm = plt.Normalize(z.min(), z.max())
        cax = fig.add_axes([0.42, 0.830, 0.11, 0.02])
        cb = plt.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=cax,
            orientation="horizontal",
        )
        cb.set_label(label, labelpad=-44, fontsize=17)
        min_reward = round(float(z.min()), 1)
        max_reward = round(float(z.max()), 1)
        mid_reward = round((min_reward + max_reward) / 2, 1)

        cb.ax.set_xticks(
            [
                min_reward,
                max_reward,
                mid_reward,
            ]
        )
        for ext in ("png", "pgf"):
            plt.savefig(SAVE_DIR / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    _surf(z_cost, "Economic Reward", "fig04c_revcomparisons_reward3d_capql_sorl")
    _surf(z_flex, "Flexibility Reward", "fig04d_revcomparisons_reward3d_capql_sorl")


# Figure 05


def figure05_violin():
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 11.5,
        }
    )
    algs = [
        "sac",
        "sac_curtailment_scaled",
        "tqc",
        "tqc_curtailment_scaled",
        "capql",
        "baseline_voltage",
    ]
    labels = ["SAC", "SAC Scaled", "TQC", "TQC Scaled", "CAPQL", "RBC"]
    base = plt.get_cmap("tab10").colors
    cmap_alg = {
        "sac": base[0],
        "sac_curtailment_scaled": _darken(base[0]),
        "tqc": base[2],
        "tqc_curtailment_scaled": _darken(base[2]),
        "capql": base[4],
        "baseline_voltage": base[5],
    }
    reward_palette = {"Curtailment": base[0], "Voltage": base[1], "Total": "#D3D3D3"}

    data = {
        a: pd.read_csv(BASE_PATH / "feeder-voltage" / "evaluations" / f"eval_{a}.csv")
        for a in algs
    }

    # Voltage violin values
    v_vals, v_labels = [], []
    for alg in algs:
        df = data[alg]
        for d in range(1, 12):
            ep = df[(df.index < d * 96 * 7) & (df.index >= (d - 1) * 96 * 7)]
            for bus in range(6):
                v_vals.extend(ep[f"voltage_{bus}"])
                v_labels.extend([alg] * len(ep))

    fig, ax = plt.subplots(2, 1, figsize=(6.45, 6.0), sharex=True)
    ax[0].grid(False)
    ax[1].grid(False)
    sns.violinplot(
        x=v_labels,
        y=v_vals,
        palette=cmap_alg,
        hue=v_labels,
        ax=ax[0],
        inner="quart",
        inner_kws={"color": "black"},
        legend=False,
    )
    ax[0].set_ylabel("Bus Voltage [p.u.]")
    ax[0].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=False)
    ax[0].set_ylim(0.87, 1.18)
    ax[0].set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax[0].axhline(0.90, linestyle="--", color="red", linewidth=1)
    ax[0].axhline(1.00, linestyle=":", color="black", linewidth=1)
    ax[0].axhline(1.10, linestyle="--", color="red", linewidth=1)
    ax[0].text(4.7, 1.115, "Damage", color="red", va="center")
    ax[0].text(4.7, 0.885, "Damage", color="red", va="center")

    alg_patches = [
        Patch(color=cmap_alg[alg], label=labels[idx]) for idx, alg in enumerate(algs)
    ]

    ax[0].legend(
        alg_patches,
        labels,
        loc="upper left",
        ncol=3,
        frameon=True,
        labelspacing=0.3,
        columnspacing=0.4,
        handletextpad=0.2,
    )

    # Reward violin
    reward_rows = []
    for alg in algs:
        df = data[alg]
        for day in range(1, 12):
            ep = df[(df.index < day * 96 * 7) & (df.index >= (day - 1) * 96 * 7)]
            reward_rows.extend(
                [
                    {
                        "Algorithm": alg,
                        "Reward Type": "Curtailment",
                        "Reward": ep["reward_curtailment"].sum(),
                    },
                    {
                        "Algorithm": alg,
                        "Reward Type": "Voltage",
                        "Reward": ep["reward_voltage"].sum(),
                    },
                    {
                        "Algorithm": alg,
                        "Reward Type": "Total",
                        "Reward": df["cumulative_reward"].iloc[(day * 96 * 7) - 1],
                    },
                ]
            )
    r_df = pd.DataFrame(reward_rows)
    r_df["Algorithm"] = pd.Categorical(r_df["Algorithm"], categories=algs, ordered=True)

    sns.violinplot(
        data=r_df,
        x="Algorithm",
        y="Reward",
        hue="Reward Type",
        palette=reward_palette,
        ax=ax[1],
        inner="quart",
        inner_kws={"color": "black"},
        linewidth=1,
        cut=0,
        legend=False,
    )
    ax[1].set_xlabel("")
    ax[1].set_ylabel("Reward Distribution")
    ax[1].set_ylim(-550, 0)
    #    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels)
    reward_patches = [
        Patch(color=reward_palette[r], label=r)
        for r in ["Curtailment", "Voltage", "Total"]
    ]
    ax[1].legend(
        reward_patches,
        ["Curtailment", "Voltage", "Total"],
        title="Reward Type",
        loc="lower left",
        ncol=3,
        frameon=True,
        labelspacing=0.3,
        columnspacing=0.4,
        handletextpad=0.2,
    )

    fig.align_ylabels(ax[:])
    plt.subplots_adjust(hspace=0.05)
    for ext in ("png", "pgf"):
        plt.savefig(SAVE_DIR / f"fig05_voltage_reward.{ext}", bbox_inches="tight")
    plt.close(fig)


def _prepare_voltage_transformer(algs: list[str], hard: bool):
    tag = "hard_eval" if hard else "eval"
    rows_v, rows_t = [], []
    groups = [
        ("Combined", BASE_PATH / "feeder-combined" / "evaluations"),
        ("Economic", BASE_PATH / "feeder-economic" / "evaluations"),
        ("Shielding", BASE_PATH / "feeder-economic" / "evaluations" / "vde-shielding"),
    ]
    for alg in algs:
        for name, base in groups:
            cand = [base / f"{tag}_{alg}.csv", base / f"{alg}.csv"]
            fn = next((p for p in cand if p.exists()), None)
            if fn is None:
                continue
            df = pd.read_csv(fn)
            rows_t.extend([(alg, name, v) for v in df["trafo_loading_0"]])
            for i in range(6):
                col = f"voltage_{i}"
                if col in df:
                    rows_v.extend([(alg, name, v) for v in df[col]])
    t_df = pd.DataFrame(rows_t, columns=["Algorithm", "Dataset", "Transformer Loading"])
    v_df = pd.DataFrame(rows_v, columns=["Algorithm", "Dataset", "Bus Voltage"])
    for d in (t_df, v_df):
        d["Algorithm"] = d["Algorithm"].str.upper()
        d["Dataset"] = pd.Categorical(
            d["Dataset"], categories=["Economic", "Shielding", "Combined"], ordered=True
        )
        # drop uncontrolled warm‑up: every 672‑nd row
        drop_idx = []
        for _, g in d.groupby(["Algorithm", "Dataset"], observed=True):
            drop_idx.extend(g.index[::672])
        d.drop(index=drop_idx, inplace=True)
    return v_df.reset_index(drop=True), t_df.reset_index(drop=True)


#  Figure 06


def figure06_soft():
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 11.5,
        }
    )

    algs = ["A2C", "DDPG", "SAC", "TD3", "TQC", "TRPO"]

    v_df, t_df = _prepare_voltage_transformer([a.lower() for a in algs], hard=False)

    fig, ax = plt.subplots(2, 1, figsize=(6.45, 6.0), sharex=True)
    ax[0].grid(False)
    ax[1].grid(False)

    sns.violinplot(
        data=v_df,
        x="Algorithm",
        y="Bus Voltage",
        hue="Dataset",
        palette="bright",
        inner="quart",
        inner_kws={"color": "black"},
        ax=ax[0],
        bw_method=0.2,
        cut=0,
    )
    ax[0].set(ylabel="Bus Voltage [p.u.]", xlabel="")
    ax[0].set_ylim(0.87, 1.18)
    ax[0].set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax[0].axhline(0.90, linestyle="--", color="red", linewidth=1)
    ax[0].axhline(1.00, linestyle=":", color="black", linewidth=1)
    ax[0].axhline(1.10, linestyle="--", color="red", linewidth=1)
    ax[0].text(4.7, 1.115, "Damage", color="red", va="center")
    ax[0].text(4.7, 0.885, "Damage", color="red", va="center")

    sns.violinplot(
        data=t_df,
        x="Algorithm",
        y="Transformer Loading",
        hue="Dataset",
        palette="bright",
        inner="quart",
        inner_kws={"color": "black"},
        ax=ax[1],
        bw_method=0.2,
        cut=0,
    )
    ax[1].set(ylabel="Transformer Loading [\\%]", xlabel="")
    ax[1].set_ylim(-10, 220)
    ax[1].set_yticks([0, 25, 50, 75, 100, 125, 150, 180, 200])
    ax[1].axhline(100, ls="--", color="black")
    ax[1].text(-0.4, 110.0, "Rated Power", color="black", va="center")
    ax[1].axhline(150, linestyle="--", color="grey", linewidth=1)
    ax[1].text(-0.4, 141.0, "Normal cyclic loading", color="black", va="center")
    ax[1].axhline(180, linestyle="--", color="grey", linewidth=1.5)
    ax[1].text(-0.4, 170.0, "Long-term loading", color="black", va="center")
    ax[1].axhline(200, linestyle="--", color="red", linewidth=1)
    ax[1].text(-0.4, 190.0, "Short-term loading", color="black", va="center")
    ax[1].text(4.7, 210.0, "Damage", color="red", va="center")
    ax[1].set_yticks([0, 25, 50, 75, 100, 125, 150, 180, 200])
    ax[1].set_ylim(-10, 220)

    ax[1].get_legend().remove()
    ax[0].legend(
        title="RL Control Mode",
        loc="upper left",
        frameon=True,
        ncol=3,
        labelspacing=0.3,
        columnspacing=0.4,
        handletextpad=0.2,
    )
    fig.align_ylabels(ax[:])
    plt.subplots_adjust(hspace=0.05)
    for ext in ("png", "pgf"):
        plt.savefig(
            SAVE_DIR / f"fig06_voltage_transformer_soft.{ext}", bbox_inches="tight"
        )
    plt.close(fig)


# Figure 08


def figure08_hard():
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 11.5,
        }
    )

    algs = ["A2C", "DDPG", "SAC", "TD3", "TQC", "TRPO"]

    v_df, t_df = _prepare_voltage_transformer([a.lower() for a in algs], hard=True)

    fig, ax = plt.subplots(2, 1, figsize=(6.45, 6.0), sharex=True)
    ax[0].grid(False)
    ax[1].grid(False)
    sns.violinplot(
        data=v_df,
        x="Algorithm",
        y="Bus Voltage",
        hue="Dataset",
        palette="bright",
        inner="quart",
        inner_kws={"color": "black"},
        ax=ax[0],
        bw_method=0.2,
        cut=0,
    )
    ax[0].set(ylabel="Bus Voltage [p.u.]", xlabel="")
    ax[0].set_ylim(0.87, 1.18)
    ax[0].set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax[0].axhline(0.90, linestyle="--", color="red", linewidth=1)
    ax[0].axhline(1.10, linestyle="--", color="red", linewidth=1)
    ax[0].axhline(1.00, linestyle=":", color="black", linewidth=1)

    ax[0].text(4.7, 1.115, "Damage", color="red", va="center")
    ax[0].text(4.7, 0.885, "Damage", color="red", va="center")

    sns.violinplot(
        data=t_df,
        x="Algorithm",
        y="Transformer Loading",
        hue="Dataset",
        palette="bright",
        inner="quart",
        inner_kws={"color": "black"},
        ax=ax[1],
        bw_method=0.2,
        cut=0,
    )
    ax[1].set(ylabel="Transformer Loading [\\%]", xlabel="")
    ax[1].set_ylim(-10, 220)
    ax[1].set_yticks([0, 25, 50, 75, 85, 100, 125, 150, 180, 200])
    ax[1].axhline(100, linestyle="--", color="black", linewidth=1)
    ax[1].text(-0.4, 108.0, "Rated Power", color="black", va="center")
    ax[1].axhline(150, linestyle="--", color="grey", linewidth=1)
    ax[1].text(-0.4, 141.0, "Normal cyclic loading", color="black", va="center")
    ax[1].axhline(180, linestyle="--", color="grey", linewidth=1.5)
    ax[1].text(-0.4, 170.0, "Long-term loading", color="black", va="center")
    ax[1].axhline(200, linestyle="--", color="red", linewidth=1)
    ax[1].text(-0.4, 190.0, "Short-term loading", color="black", va="center")
    ax[1].text(4.7, 210.0, "Damage", color="red", va="center")
    ax[1].axhline(85, color="blue", linewidth=2)
    ax[1].text(
        2.1,
        108.0,
        "Transformer Constraint Enforced",
        color="blue",
        va="center",
        fontweight="bold",
    )

    ax[1].get_legend().remove()
    ax[0].legend(
        title="RL Control Mode",
        loc="upper left",
        frameon=True,
        ncol=3,
        labelspacing=0.3,
        columnspacing=0.4,
        handletextpad=0.2,
    )
    fig.align_ylabels(ax[:])
    plt.subplots_adjust(hspace=0.05)
    for ext in ("png", "pgf"):
        plt.savefig(
            SAVE_DIR / f"fig08_voltage_transformer_hard.{ext}", bbox_inches="tight"
        )
    plt.close(fig)


#  Main


def main():
    print(">> Generating all figures for the paper. Please wait ...")
    figure03_training_curves()
    figure04a_bar()
    figure04b_pareto()
    figure04c_04d_3d()
    figure05_violin()
    figure06_soft()
    figure08_hard()
    print(f">> All figures successfully saved to: {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
