#!/usr/bin/env python3
"""Run 173 K EMI-BF4 MD and plot B-F bond-distance distributions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "scripts" / "simulation" / "inputs" / "EMIBF4.dat"
OUTDIR = ROOT / "results" / "bf4_173k_distribution"

BF_BONDS = {
    "B1-F1": (19, 20),
    "B1-F2": (19, 21),
    "B1-F3": (19, 22),
    "B1-F4": (19, 23),
}

DFT_BF = {
    "B1-F1": 1.364,
    "B1-F2": 1.365,
    "B1-F3": 1.378,
    "B1-F4": 1.703,
}

EXPERIMENT_BF = {
    "B1-F1": 1.376,
    "B1-F2": 1.386,
    "B1-F3": 1.391,
    "B1-F4": 1.399,
}


class CenteredCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calculator, cell_A=35.0):
        super().__init__()
        self.base_calculator = base_calculator
        self.cell_A = cell_A

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        centered = atoms.copy()
        positions = centered.positions
        midpoint = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
        centered.positions = positions - midpoint
        centered.set_cell(np.eye(3) * self.cell_A)
        centered.set_pbc(False)
        self.base_calculator.calculate(centered, properties, all_changes)
        self.results = {
            "energy": self.base_calculator.results.get("energy", 0.0),
            "forces": self.base_calculator.results.get("forces", np.zeros((len(atoms), 3))),
        }


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 15,
            "font.weight": "bold",
            "legend.fontsize": 11,
            "axes.linewidth": 2.0,
            "lines.linewidth": 2.2,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.major.size": 6,
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.size": 3,
            "xtick.major.width": 1.6,
            "xtick.minor.width": 1.2,
            "xtick.direction": "in",
            "ytick.major.size": 6,
            "ytick.minor.size": 3,
            "ytick.major.width": 1.6,
            "ytick.minor.width": 1.2,
            "ytick.direction": "in",
            "xtick.major.top": True,
            "xtick.minor.top": True,
            "ytick.major.right": True,
            "ytick.minor.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def isolated_start(cell_A=35.0):
    atoms = read(INPUT, format="lammps-data", atom_style="full")
    atoms.set_cell(np.eye(3) * cell_A)
    atoms.center()
    atoms.set_pbc(False)
    return atoms


def make_calculator(model: str, device: str, dtype: str):
    if model == "MACE-medium":
        from mace.calculators import mace_mp

        calc = mace_mp(model="medium", device=device, default_dtype=dtype)
    elif model == "MACE-POLAR-1":
        from mace.calculators import mace_polar

        calc = mace_polar(model="polar-1-m", device=device, default_dtype=dtype)
    else:
        raise ValueError(model)
    return CenteredCalculator(calc)


def optimize(atoms, model: str, device: str, dtype: str, fmax: float, max_steps: int):
    atoms = atoms.copy()
    atoms.calc = make_calculator(model, device, dtype)
    tag = model.replace("-", "_")
    opt = BFGS(
        atoms,
        trajectory=str(OUTDIR / f"{tag}_opt.traj"),
        logfile=str(OUTDIR / f"{tag}_opt.log"),
    )
    opt.run(fmax=fmax, steps=max_steps)
    write(OUTDIR / f"{tag}_optimized.xyz", atoms)
    return atoms


def collect_bond_distances(atoms, model: str, args):
    rng_seed = args.seed + (0 if model == "MACE-medium" else 1000)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.default_rng(rng_seed))
    Stationary(atoms)
    ZeroRotation(atoms)
    atoms.calc = make_calculator(model, args.device, args.dtype)

    dyn = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_K,
        friction=args.friction_per_fs / units.fs,
        trajectory=None,
        logfile=None,
    )

    equil_steps = int(round(args.equil_ps * 1000 / args.timestep_fs))
    prod_steps = int(round(args.production_ps * 1000 / args.timestep_fs))
    sample_interval = max(1, int(round(args.sample_interval_fs / args.timestep_fs)))

    if equil_steps:
        dyn.run(equil_steps)

    rows = []
    for step in range(prod_steps + 1):
        if step % sample_interval == 0:
            time_ps = step * args.timestep_fs / 1000
            for bond, (i, j) in BF_BONDS.items():
                rows.append(
                    {
                        "method": model,
                        "bond": bond,
                        "time_ps": time_ps,
                        "distance_A": atoms.get_distance(i, j, mic=False),
                    }
                )
        if step < prod_steps:
            dyn.run(1)

    write(OUTDIR / f"{model.replace('-', '_')}_173K_final.xyz", atoms)
    return pd.DataFrame(rows)


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    grouped = samples.groupby(["method", "bond"])["distance_A"]
    summary = grouped.agg(["mean", "std", "min", "max"]).reset_index()
    summary["dft_A"] = summary["bond"].map(DFT_BF)
    summary["experiment_A"] = summary["bond"].map(EXPERIMENT_BF)
    summary["mean_abs_error_vs_dft_A"] = (summary["mean"] - summary["dft_A"]).abs()
    summary["mean_abs_error_vs_experiment_A"] = (summary["mean"] - summary["experiment_A"]).abs()
    return summary


def plot_distributions(samples: pd.DataFrame, summary: pd.DataFrame) -> None:
    configure_plot_style()
    colors = {"MACE-medium": "#0072B2", "MACE-POLAR-1": "#D55E00"}
    bonds = list(BF_BONDS)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharey=True)

    for ax, bond in zip(axes.flat, bonds):
        x_max = 1.74 if bond == "B1-F4" else 1.52
        bins = np.linspace(1.30, x_max, 42)
        for method in ["MACE-medium", "MACE-POLAR-1"]:
            values = samples.loc[(samples["method"] == method) & (samples["bond"] == bond), "distance_A"]
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.4,
                color=colors[method],
                label=method,
            )
            stats = summary[(summary["method"] == method) & (summary["bond"] == bond)].iloc[0]
            ax.axvline(stats["mean"], color=colors[method], linestyle="-", linewidth=1.5, alpha=0.75)

        ax.axvline(DFT_BF[bond], color="black", linestyle="--", linewidth=2.0, label="DFT")
        ax.axvline(EXPERIMENT_BF[bond], color="#009E73", linestyle=":", linewidth=2.4, label="Experiment")
        ax.set_title(bond)
        ax.set_xlim(1.30, x_max)
        ax.minorticks_on()
        ax.text(
            0.04,
            0.92,
            f"DFT {DFT_BF[bond]:.3f} A\nExp {EXPERIMENT_BF[bond]:.3f} A",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )

    axes[1, 0].set_xlabel("B-F distance (A)")
    axes[1, 1].set_xlabel("B-F distance (A)")
    axes[0, 0].set_ylabel("Probability density")
    axes[1, 0].set_ylabel("Probability density")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.00))
    fig.suptitle("173 K BF4 Bond-Length Distributions in EMI-BF4", y=0.985, fontsize=21, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(OUTDIR / "bf4_173K_bond_distribution.png", dpi=300)
    fig.savefig(OUTDIR / "bf4_173K_bond_distribution.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--temperature-K", type=float, default=173.0)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--equil-ps", type=float, default=1.0)
    parser.add_argument("--production-ps", type=float, default=5.0)
    parser.add_argument("--sample-interval-fs", type=float, default=5.0)
    parser.add_argument("--friction-per-fs", type=float, default=0.01)
    parser.add_argument("--fmax", type=float, default=0.03)
    parser.add_argument("--opt-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    start = isolated_start()
    all_samples = []
    for model in ["MACE-medium", "MACE-POLAR-1"]:
        print(f"Optimizing {model}...")
        optimized = optimize(start, model, args.device, args.dtype, args.fmax, args.opt_steps)
        print(f"Running {args.temperature_K:g} K MD for {model}...")
        all_samples.append(collect_bond_distances(optimized, model, args))

    samples = pd.concat(all_samples, ignore_index=True)
    summary = summarize(samples)
    samples.to_csv(OUTDIR / "bf4_173K_bond_distance_samples.csv", index=False)
    summary.to_csv(OUTDIR / "bf4_173K_bond_distance_summary.csv", index=False)
    plot_distributions(samples, summary)

    print(summary.round(4).to_string(index=False))
    print(f"Wrote outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
