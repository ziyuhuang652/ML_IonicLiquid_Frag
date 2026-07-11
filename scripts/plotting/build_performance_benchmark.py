#!/usr/bin/env python3
"""Build the README performance benchmark figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "benchmarks" / "resource_benchmark.csv"
OUT = ROOT / "assets" / "performance_benchmark"


def format_time(hours: float) -> str:
    if hours >= 24:
        return f"{hours / 24:.0f} days"
    if hours >= 1:
        return f"{hours:.1f} h"
    if hours >= 1 / 60:
        return f"{hours * 60:.2f} min"
    return f"{hours * 3600:.2f} s"


def main() -> None:
    data = pd.read_csv(DATA)
    order = ["DFT/QM-MM", "MACE-POLAR-1", "MACE-medium", "ReaxFF"]
    labels = ["DFT/MM", "MACE-POLAR-1", "MACE-medium", "ReaxFF"]
    colors = ["#4D4D4D", "#D55E00", "#0072B2", "#009E73"]
    data = data.set_index("method").loc[order].reset_index()
    hours = data["wall_time_hours_per_2ps_trajectory"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bars = ax.bar(labels, hours, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Wall time per 2 ps trajectory (hours, log scale)")
    ax.set_title("Wall-collision simulation performance benchmark")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, value in zip(bars, hours):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.35,
            format_time(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.text(
        0.01,
        0.01,
        "DFT/MM: reported upper-bound runtime; ReaxFF/MACE: local projected or measured benchmarks.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
