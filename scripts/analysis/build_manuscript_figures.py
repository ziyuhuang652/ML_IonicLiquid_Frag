#!/usr/bin/env python3
"""Build the evidence-led figures used by the Acta Astronautica manuscript."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import covalent_radii
from ase.io import read
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


PACKAGE = Path(__file__).resolve().parents[2]
DATA = PACKAGE / "data"
PROCESSED = DATA / "processed"
FIGURES = PACKAGE / "manuscript" / "figures"
ENERGIES = [10, 20, 30, 40, 50, 75, 100]
SELECTED_ENERGIES = [10, 40, 100]

COLORS = {
    "DFT-MD": "#4D4D4D",
    "ReaxFF": "#009E73",
    "MACE-medium": "#0072B2",
    "MACE-polar": "#D55E00",
}
DISPLAY = {
    "DFT-MD": "DFT/MM",
    "ReaxFF": "ReaxFF rerun",
    "MACE-medium": "MACE-medium*",
    "MACE-polar": "MACE-POLAR-1",
}
ELEMENT_COLORS = {
    "H": "#D9D9D9",
    "B": "#E69F00",
    "C": "#4D4D4D",
    "N": "#0072B2",
    "F": "#009E73",
}
ELEMENT_SIZES = {"H": 24, "B": 72, "C": 62, "N": 68, "F": 66}
AU_COLOR = "#D8A319"
AU_EDGE = "#8A6500"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def draw_molecule(ax: plt.Axes, atoms, title: str, show_wall: bool) -> None:
    """Draw a y-z projection with connectivity inferred from covalent radii."""
    positions = atoms.positions
    y = positions[:, 1]
    z = positions[:, 2]

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            cutoff = 1.25 * (covalent_radii[atoms.numbers[i]] + covalent_radii[atoms.numbers[j]])
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= cutoff:
                symbols = {atoms.symbols[i], atoms.symbols[j]}
                is_hf = symbols == {"H", "F"}
                ax.plot(
                    [y[i], y[j]],
                    [z[i], z[j]],
                    color="#CC3311" if is_hf else "#8A8A8A",
                    linewidth=2.1 if is_hf else 0.8,
                    zorder=1,
                )

    order = np.argsort(positions[:, 0])
    for idx in order:
        symbol = atoms.symbols[idx]
        ax.scatter(
            y[idx],
            z[idx],
            s=ELEMENT_SIZES[symbol],
            color=ELEMENT_COLORS[symbol],
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )

    if show_wall:
        ax.axhline(0, color="#A67C00", linewidth=3)
        ax.fill_between(
            [y.min() - 2, y.max() + 2],
            -1.4,
            0,
            color="#F0E1A6",
            alpha=0.8,
            zorder=0,
        )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Lateral coordinate (A)")
    ax.set_ylabel("Surface-normal coordinate (A)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)


def build_collision_sequence() -> None:
    path = (
        DATA
        / "raw/mace_polar_42/wall_EMIBF4_E0040eV_Odipole_nx"
        / "wall_EMIBF4_E0040eV_Odipole_nx.xyz"
    )
    frames = read(path, ":")
    selected = [(0, "Initial state\n0 fs"), (100, "Reactive contact\n500 fs"), (200, "Scattered products\n1000 fs")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    for label, (ax, (index, title)) in zip("ABC", zip(axes, selected)):
        draw_molecule(ax, frames[index], title, show_wall=index < 160)
        add_panel_label(ax, label)

    legend = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
               markeredgecolor="white", label=element, markersize=6)
        for element, color in ELEMENT_COLORS.items()
    ]
    legend.append(Line2D([0], [0], color="#CC3311", linewidth=2, label="H--F bond"))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.27, wspace=0.42)
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.01))
    export(fig, "fig1_collision_sequence")


def draw_collision_render(
    ax: plt.Axes,
    atoms,
    view: tuple[float, float],
    title: str,
    show_velocity_label: bool,
) -> None:
    slab = atoms[:256]
    projectile = atoms[256:]
    slab_top = slab.positions[:, 2].max()
    top_layer = slab[slab.positions[:, 2] > slab_top - 1.0]
    pos = projectile.positions
    symbols = projectile.get_chemical_symbols()

    x_min, x_max = top_layer.positions[:, 0].min(), top_layer.positions[:, 0].max()
    y_min, y_max = top_layer.positions[:, 1].min(), top_layer.positions[:, 1].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min - 1.2, x_max + 1.2, 2),
        np.linspace(y_min - 1.2, y_max + 1.2, 2),
    )
    zz = np.full_like(xx, slab_top - 0.08)
    ax.plot_surface(xx, yy, zz, color="#F2D36B", alpha=0.18, shade=False, linewidth=0)

    ax.scatter(
        top_layer.positions[:, 0],
        top_layer.positions[:, 1],
        top_layer.positions[:, 2],
        c=AU_COLOR,
        s=38,
        alpha=0.62,
        edgecolors=AU_EDGE,
        linewidths=0.25,
        depthshade=True,
        zorder=0,
    )

    for i in range(len(projectile)):
        for j in range(i + 1, len(projectile)):
            cutoff = 1.25 * (
                covalent_radii[projectile.numbers[i]]
                + covalent_radii[projectile.numbers[j]]
            )
            if np.linalg.norm(pos[i] - pos[j]) <= cutoff:
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    [pos[i, 2], pos[j, 2]],
                    color="#777777",
                    linewidth=1.15,
                    alpha=0.75,
                    zorder=4,
                )

    draw_order = np.argsort(pos[:, 2])
    for idx in draw_order:
        symbol = symbols[idx]
        ax.scatter(
            [pos[idx, 0]],
            [pos[idx, 1]],
            [pos[idx, 2]],
            c=ELEMENT_COLORS[symbol],
            s=ELEMENT_SIZES[symbol] * 1.45,
            edgecolors="white",
            linewidths=0.45,
            depthshade=True,
            zorder=8,
        )

    center = pos.mean(axis=0)
    arrow_start = center + np.array([0.0, 0.0, 5.0])
    ax.quiver(
        arrow_start[0],
        arrow_start[1],
        arrow_start[2],
        0.0,
        0.0,
        -4.5,
        color="#D55E00",
        linewidth=2.4,
        arrow_length_ratio=0.22,
        normalize=False,
    )
    if show_velocity_label:
        ax.text(
            arrow_start[0] + 0.7,
            arrow_start[1] + 0.7,
            arrow_start[2] - 1.6,
            r"$v_{\mathrm{COM}}$",
            color="#D55E00",
            fontsize=10,
            fontweight="bold",
        )

    span = max(x_max - x_min, y_max - y_min) / 2
    ax.set_xlim(center[0] - span * 0.72, center[0] + span * 0.72)
    ax.set_ylim(center[1] - span * 0.72, center[1] + span * 0.72)
    ax.set_zlim(slab_top - 2.0, max(pos[:, 2].max() + 6.5, slab_top + 16))
    ax.set_box_aspect((1.0, 1.0, 0.72))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_title(title, pad=4)
    ax.set_axis_off()


def build_setup_figure() -> None:
    path = (
        DATA
        / "raw/mace_medium_42/run_E10_Y-90_Z-90"
        / "E10_Y-90_Z-90.traj"
    )
    atoms = read(path, 0)
    slab = atoms[:256]
    projectile = atoms[256:]
    slab_top = slab.positions[:, 2].max()
    projectile_bottom = projectile.positions[:, 2].min()

    fig = plt.figure(figsize=(7.2, 3.2))
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    draw_collision_render(
        axes[0],
        atoms,
        view=(24, -58),
        title="EMI--BF$_4$ approaching Au(111)",
        show_velocity_label=True,
    )
    add_panel_label(axes[0], "A")
    draw_collision_render(
        axes[1],
        atoms,
        view=(7, -90),
        title="Side view and assigned impact velocity",
        show_velocity_label=False,
    )
    axes[1].text2D(
        0.03,
        0.08,
        f"Initial clearance: {projectile_bottom - slab_top:.1f} A\n"
        r"$v_{\mathrm{COM}}=\sqrt{2E/m}$",
        transform=axes[1].transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.88, "pad": 2},
    )
    add_panel_label(axes[1], "B")
    legend = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color,
               markeredgecolor="white", label=element, markersize=6)
        for element, color in ELEMENT_COLORS.items()
    ]
    legend.append(
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=AU_COLOR,
               markeredgecolor=AU_EDGE, label="Au", markersize=6)
    )
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.00))
    fig.subplots_adjust(left=0.00, right=0.99, top=0.90, bottom=0.15, wspace=0.00)
    export(fig, "fig1_collision_setup")


def build_mass_spectra() -> None:
    spectra = pd.read_csv(PROCESSED / "mass_spectra_standardized.csv")
    methods = ["DFT-MD", "ReaxFF", "MACE-medium", "MACE-polar"]
    fig, axes = plt.subplots(
        len(SELECTED_ENERGIES),
        len(methods),
        figsize=(7.2, 4.25),
        sharex=True,
        constrained_layout=True,
    )

    for column, method in enumerate(methods):
        for row, energy in enumerate(SELECTED_ENERGIES):
            ax = axes[row, column]
            subset = spectra[
                (spectra.method == method) & (spectra.energy_eV == energy)
            ].sort_values("mass_amu")
            values = subset.probability.to_numpy(float)
            relative = values / values.max() if len(values) and values.max() > 0 else values
            ax.vlines(
                subset.mass_amu,
                0,
                relative,
                color=COLORS[method],
                linewidth=1.05,
            )
            ax.set_ylim(0, 1.08)
            ax.set_xlim(0, 205)
            ax.set_yticks([0, 1])
            ax.text(
                0.97,
                0.78,
                f"{energy} eV",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=7,
            )
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(DISPLAY[method], fontweight="bold")
            if column == 0 and row == 1:
                ax.set_ylabel("Relative occurrence within each spectrum")
            if row == len(SELECTED_ENERGIES) - 1:
                ax.set_xlabel("Fragment mass (amu)")
            else:
                ax.tick_params(labelbottom=False)

    for column, label in enumerate("ABCD"):
        add_panel_label(axes[0, column], label)
    fig.text(0.5, -0.012, "Peak heights are normalized within each method-energy panel.", ha="center", fontsize=7)
    export(fig, "fig1_selected_mass_spectra")


def boolean_map(rows: pd.DataFrame, value: str, orientations: list[str]) -> np.ndarray:
    result = np.full((len(ENERGIES), len(orientations)), np.nan)
    for i, energy in enumerate(ENERGIES):
        for j, orientation in enumerate(orientations):
            match = rows[
                (rows.energy_eV == energy)
                & (rows.orientation_native == orientation)
            ]
            if not match.empty:
                result[i, j] = float(bool(match.iloc[0][value]))
    return result


def build_outcome_maps() -> None:
    wall = pd.read_csv(PROCESSED / "wall_cases_normalized.csv")
    hf = pd.read_csv(PROCESSED / "hf_trajectory_occurrence.csv")
    reax = pd.read_csv(PROCESSED / "reaxff_case_summary.csv").rename(
        columns={"orientation": "orientation_native"}
    )
    reax["model"] = "ReaxFF"
    reax["any_HF"] = False

    hf_lookup = hf.set_index(["model", "case_id"])["any_HF"]
    wall["any_HF"] = [
        bool(hf_lookup.get((row.model, row.case_id), False))
        for row in wall.itertuples()
    ]
    combined = pd.concat(
        [
            wall[["model", "energy_eV", "orientation_native", "covalent_breakup", "any_HF"]],
            reax[["model", "energy_eV", "orientation_native", "covalent_breakup", "any_HF"]],
        ],
        ignore_index=True,
    )

    models = ["MACE-medium", "MACE-polar", "ReaxFF"]
    orientation_sets = {
        "MACE-medium": ["Y-90_Z-90", "Y-90_Z90", "Y45_Z-90", "Y45_Z90", "Y180_Z-90", "Y180_Z90"],
        "MACE-polar": ["dipole_px", "dipole_nx", "dipole_py", "dipole_ny", "dipole_pz", "dipole_nz"],
        "ReaxFF": ["dipole_px", "dipole_nx", "dipole_py", "dipole_ny", "dipole_pz", "dipole_nz"],
    }
    orientation_labels = {
        "MACE-medium": ["Y-90/Z-90", "Y-90/Z90", "Y45/Z-90", "Y45/Z90", "Y180/Z-90", "Y180/Z90"],
        "MACE-polar": ["+x", "-x", "+y", "-y", "+z", "-z"],
        "ReaxFF": ["+x", "-x", "+y", "-y", "+z", "-z"],
    }
    cmap = ListedColormap(["#F2F2F2", "#D55E00"])
    regime_cmap = ListedColormap(["#56B4E9", "#009E73", "#E69F00"])
    fig = plt.figure(figsize=(7.2, 3.25))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.65, 2, 2, 2])
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]

    dft_ax = axes[0]
    dft_regimes = np.array([0, 0, 1, 1, 2, 2, 2])[:, None]
    dft_ax.imshow(dft_regimes, cmap=regime_cmap, vmin=0, vmax=2, aspect="auto")
    dft_ax.set_xticks([0], ["Pooled\nregime"])
    dft_ax.set_yticks(range(7), ENERGIES)
    dft_ax.set_ylabel("Impact energy (eV)")
    dft_ax.set_title("DFT/MM", fontweight="bold")
    for i, text in enumerate(["I", "I", "N", "N", "C", "C", "C"]):
        dft_ax.text(0, i, text, ha="center", va="center", color="white", fontweight="bold")
    add_panel_label(dft_ax, "A")

    for column, model in enumerate(models, start=1):
        subset = combined[combined.model == model]
        orientations = orientation_sets[model]
        ax = axes[column]
        matrix = boolean_map(subset, "covalent_breakup", orientations)
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(6), orientation_labels[model], rotation=45, ha="right")
        ax.set_yticks(range(7), [])
        ax.set_xlabel("Projectile orientation")
        ax.set_title(DISPLAY[model], fontweight="bold")
        ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(7):
            for j in range(6):
                ax.text(
                    j,
                    i,
                    "1" if matrix[i, j] == 1 else "0",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] == 1 else "#666666",
                    fontsize=7,
                )
        add_panel_label(ax, chr(ord("A") + column))

    legend = [
        Line2D([0], [0], marker="s", linestyle="", color="#56B4E9",
               label="DFT ionic dissociation (I)", markersize=7),
        Line2D([0], [0], marker="s", linestyle="", color="#009E73",
               label="DFT neutralization (N)", markersize=7),
        Line2D([0], [0], marker="s", linestyle="", color="#E69F00",
               label="DFT covalent fragmentation (C)", markersize=7),
        Line2D([0], [0], marker="s", linestyle="", color="#F2F2F2",
               markeredgecolor="#999999", label="Not observed", markersize=7),
        Line2D([0], [0], marker="s", linestyle="", color="#D55E00",
               label="Observed", markersize=7),
    ]
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.34, wspace=0.34)
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    export(fig, "fig2_fragmentation_maps")


def build_product_families() -> None:
    dft = pd.read_csv(DATA / "literature/dft_energy_resolved_results.csv")
    products = pd.read_csv(PROCESSED / "wall_products.csv")
    reax = pd.read_csv(PROCESSED / "reaxff_case_summary.csv")
    methods = ["DFT-MD", "ReaxFF", "MACE-medium", "MACE-polar"]
    selected_products = ["H", "CH3", "F", "HF", "BF2", "BF3", "BF4", "C2H2", "C2H3N", "C6H10N2", "C6H11N2"]
    label_map = {
        "CH3": r"CH$_3$",
        "BF2": r"BF$_2$",
        "BF3": r"BF$_3$",
        "BF4": r"BF$_4$",
        "C2H2": r"C$_2$H$_2$",
        "C2H3N": r"C$_2$H$_3$N",
        "C6H10N2": r"C$_6$H$_{10}$N$_2$",
        "C6H11N2": r"C$_6$H$_{11}$N$_2$",
    }
    matrices = {}
    for method in methods:
        matrix = np.zeros((len(selected_products), len(SELECTED_ENERGIES)))
        for column, energy in enumerate(SELECTED_ENERGIES):
            if method == "DFT-MD":
                value = dft.loc[dft.energy_eV == energy, "reported_products"].iloc[0]
                present = set(value.split(";"))
            elif method == "ReaxFF":
                values = reax.loc[reax.energy_eV == energy, "final_fragment_formulas"]
                present = set(";".join(values).split(";"))
            else:
                present = set(
                    products.loc[
                        (products.model == method) & (products.energy_eV == energy),
                        "product_formula",
                    ]
                )
            for row, product in enumerate(selected_products):
                matrix[row, column] = product in present
        matrices[method] = matrix

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.75), constrained_layout=True)
    product_cmap = ListedColormap(["#F2F2F2", "#0072B2"])
    for index, (ax, method) in enumerate(zip(axes, methods)):
        matrix = matrices[method]
        ax.imshow(matrix, cmap=product_cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(3), SELECTED_ENERGIES)
        ax.set_xlabel("Impact energy (eV)")
        ax.set_yticks(
            range(len(selected_products)),
            [label_map.get(item, item) for item in selected_products] if index == 0 else [],
        )
        ax.set_title(DISPLAY[method], fontweight="bold")
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(selected_products), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(len(selected_products)):
            for column in range(3):
                ax.text(
                    column,
                    row,
                    "●" if matrix[row, column] else "–",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] else "#777777",
                    fontsize=7,
                )
        add_panel_label(ax, chr(ord("A") + index))
    export(fig, "fig3_product_family_comparison")


def build_hf_mechanism() -> None:
    trajectory = (
        DATA
        / "raw/mace_polar_42/wall_EMIBF4_E0040eV_Odipole_nx"
        / "wall_EMIBF4_E0040eV_Odipole_nx.xyz"
    )
    frames = read(trajectory, ":")
    symbols = frames[0].get_chemical_symbols()
    hydrogens = [i for i, symbol in enumerate(symbols) if symbol == "H"]
    fluorines = [i for i, symbol in enumerate(symbols) if symbol == "F"]
    boron = symbols.index("B")

    _, terminal_h, terminal_f = min(
        (frames[-1].get_distance(h, f), h, f)
        for h in hydrogens
        for f in fluorines
    )
    hf_distance = [atoms.get_distance(terminal_h, terminal_f) for atoms in frames]
    associated_bf = [atoms.get_distance(boron, terminal_f) for atoms in frames]

    times = np.arange(len(frames)) * 5.0
    hf_summary = pd.read_csv(PROCESSED / "hf_energy_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    axes[0].plot(times, hf_distance, color="#CC3311", linewidth=1.4, label="H--F product pair")
    axes[0].plot(times, associated_bf, color="#0072B2", linewidth=1.2, label="B--F precursor bond")
    axes[0].axhspan(0, 1.3, color="#CC3311", alpha=0.08)
    axes[0].set(
        xlabel="Time (fs)",
        ylabel="Interatomic distance (A)",
        xlim=(0, times[-1]),
        ylim=(0, 8),
        title="Representative 40-eV MACE-POLAR-1 trajectory",
    )
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].spines[["top", "right"]].set_visible(False)
    add_panel_label(axes[0], "A")

    for model in ["MACE-medium", "MACE-polar", "ReaxFF"]:
        subset = hf_summary[hf_summary.model == model].sort_values("energy_eV")
        axes[1].plot(
            subset.energy_eV,
            subset.HF_case_fraction,
            marker={"MACE-medium": "s", "MACE-polar": "o", "ReaxFF": "^"}[model],
            color=COLORS[model],
            linewidth=1.3,
            label=DISPLAY[model],
        )
    axes[1].axvspan(30, 40, color="#E69F00", alpha=0.15, label="DFT neutralization window")
    axes[1].set(
        xlabel="Impact energy (eV)",
        ylabel="Fraction of trajectories with HF",
        xlim=(7, 103),
        ylim=(-0.03, 1.03),
        title="Trajectory-resolved HF incidence",
    )
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)
    add_panel_label(axes[1], "B")
    export(fig, "fig4_hf_mechanism_and_incidence")


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_setup_figure()
    build_mass_spectra()
    build_outcome_maps()
    build_product_families()
    print("Wrote four manuscript figures in PDF and PNG formats.")


if __name__ == "__main__":
    main()
