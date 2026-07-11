#!/usr/bin/env python3
"""Build the README performance benchmark figure."""

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmarks" / "resource_benchmark.csv"
OUT = ROOT / "assets" / "performance_benchmark"
Y_FLOOR = 1e-4


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 23,
            "font.weight": "bold",
            "legend.fontsize": 23,
            "axes.linewidth": 3.0,
            "lines.linewidth": 3.0,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.major.size": 8,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.size": 4,
            "xtick.major.width": 2,
            "xtick.minor.width": 1.5,
            "xtick.direction": "in",
            "ytick.major.size": 8,
            "ytick.minor.size": 4,
            "ytick.major.width": 2,
            "ytick.minor.width": 1,
            "ytick.direction": "in",
            "xtick.major.top": True,
            "xtick.minor.top": True,
            "ytick.major.right": True,
            "ytick.minor.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_time(hours: float) -> str:
    if hours >= 24:
        return f"{hours / 24:.0f} days"
    if hours >= 1:
        return f"{hours:.1f} h"
    if hours >= 1 / 60:
        return f"{hours * 60:.2f} min"
    return f"{hours * 3600:.2f} s"


def main() -> None:
    configure_style()
    data = pd.read_csv(DATA)
    order = ["DFT/QM-MM", "MACE-medium", "MACE-POLAR-1", "ReaxFF"]
    labels = ["DFT/MM", "MACE-\nMedium", "MACE-\nPolar", "ReaxFF"]
    colors = ["#4D4D4D", "#0072B2", "#D55E00", "#009E73"]
    chemistry = ["HF", "HF", "HF", "No HF"]
    data = data.set_index("method").loc[order].reset_index()
    hours = data["wall_time_hours_per_2ps_trajectory"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    bars = ax.bar(labels, hours, color=colors, edgecolor="black", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_ylim(Y_FLOOR, 5e3)
    ax.set_ylabel("Wall Time / 2 ps Trajectory (h)")
    ax.set_title("Performance Benchmark")
    ax.grid(False)
    ax.tick_params(axis="x", pad=10)

    for bar, value, label in zip(bars, hours, chemistry):
        y_text = 10 ** ((math.log10(max(value, Y_FLOOR)) + math.log10(Y_FLOOR)) / 2)
        is_reaxff = label == "No HF"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_text,
            label,
            ha="center",
            va="center",
            rotation=0 if is_reaxff else 90,
            fontsize=14 if is_reaxff else 22,
            fontweight="bold",
            color="white",
            clip_on=True,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.45,
            format_time(value),
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
