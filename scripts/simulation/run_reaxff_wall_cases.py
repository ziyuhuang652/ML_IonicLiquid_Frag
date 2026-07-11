#!/usr/bin/env python3
"""Generate, run, and analyze the reduced 42-case ReaxFF wall campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from ase.io import read, write


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
DEFAULT_RESULTS_DIR = ROOT / "results"
BASE_DATA = INPUT_DIR / "eq_SYST.dat"
FFIELD = INPUT_DIR / "ffield_IL.txt"
LAMMPS = Path(os.environ.get("LAMMPS_EXECUTABLE", shutil.which("lmp") or "lmp"))

ENERGIES_EV = [10, 20, 30, 40, 50, 75, 100]
ORIENTATIONS = [
    "dipole_px",
    "dipole_nx",
    "dipole_py",
    "dipole_ny",
    "dipole_pz",
    "dipole_nz",
]
TARGETS = {
    "dipole_px": np.array([1.0, 0.0, 0.0]),
    "dipole_nx": np.array([-1.0, 0.0, 0.0]),
    "dipole_py": np.array([0.0, 1.0, 0.0]),
    "dipole_ny": np.array([0.0, -1.0, 0.0]),
    "dipole_pz": np.array([0.0, 0.0, 1.0]),
    "dipole_nz": np.array([0.0, 0.0, -1.0]),
}
DFT_EULER_ANGLES = {
    "dipole_px": [0, 0],
    "dipole_nx": [180, 0],
    "dipole_py": [0, 90],
    "dipole_ny": [180, 90],
    "dipole_pz": [-90, 0],
    "dipole_nz": [90, 0],
}
TYPE_SYMBOLS = {1: "N", 2: "C", 3: "H", 4: "B", 5: "F"}
TYPE_MASSES = {1: 14.0070, 2: 12.0010, 3: 1.0080, 4: 10.811, 5: 18.9984}
FORMULA_ORDER = ["C", "H", "B", "F", "N"]
WALL_X = 69.0
DT_FS = 0.5
DEFAULT_STEPS = 4000
DUMP_INTERVAL = 100


def rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm < 1e-12:
        if dot > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, trial)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    axis = cross / norm
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    angle = math.acos(dot)
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def case_id(energy: float, orientation: str) -> str:
    return f"wall_EMIBF4_E{energy:04g}eV_O{orientation}"


def generate_case(case_dir: Path, energy: float, orientation: str, steps: int) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(BASE_DATA, format="lammps-data", atom_style="full", units="real")
    atoms.set_pbc(False)

    positions = atoms.get_positions()
    velocities = atoms.get_velocities()
    masses = atoms.get_masses()
    charges = atoms.get_initial_charges()
    center = atoms.get_center_of_mass()
    dipole = np.sum(charges[:, None] * (positions - center), axis=0)
    rotation = rotation_matrix(dipole, TARGETS[orientation])

    atoms.set_positions((positions - center) @ rotation.T + center + np.array([50.0, 16.3128, 16.3128]))
    com_velocity = np.average(velocities, axis=0, weights=masses)
    internal_velocities = (velocities - com_velocity) @ rotation.T
    impact_velocity = math.sqrt(2.0 * energy / masses.sum())
    internal_velocities[:, 0] += impact_velocity
    atoms.set_velocities(internal_velocities)

    final_dipole = np.sum(
        charges[:, None] * (atoms.get_positions() - atoms.get_center_of_mass()), axis=0
    )
    translational_velocity = np.average(atoms.get_velocities(), axis=0, weights=masses)
    translational_energy = 0.5 * masses.sum() * float(np.dot(translational_velocity, translational_velocity))
    internal_energy = atoms.get_kinetic_energy() - translational_energy

    write(
        case_dir / "initial.data",
        atoms,
        format="lammps-data",
        atom_style="full",
        units="real",
        velocities=True,
        masses=True,
        specorder=["N", "C", "H", "B", "F"],
    )
    metadata = {
        "case_id": case_dir.name,
        "energy_eV": energy,
        "orientation": orientation,
        "orientation_target": TARGETS[orientation].tolist(),
        "dft_euler_angles_phi_y_phi_z_deg": DFT_EULER_ANGLES[orientation],
        "dft_phi_y_deg": DFT_EULER_ANGLES[orientation][0],
        "dft_phi_z_deg": DFT_EULER_ANGLES[orientation][1],
        "initial_dipole": dipole.tolist(),
        "oriented_dipole": final_dipole.tolist(),
        "impact_axis": "+x",
        "impact_velocity_A_per_fs": impact_velocity * 0.09822694788464063,
        "initialized_translational_energy_eV": translational_energy,
        "retained_internal_kinetic_energy_eV": internal_energy,
        "wall_x_A": WALL_X,
        "initial_com_A": atoms.get_center_of_mass().tolist(),
        "dt_fs": DT_FS,
        "steps": steps,
        "simulation_time_fs": steps * DT_FS,
        "source_setup": (
            "Bendimerad and Petro ReaxFF model evaluated on the Laws and Petro "
            "seven-energy, six-orientation, 2 ps case matrix"
        ),
    }
    (case_dir / "initialization.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    input_text = f"""# Reduced EMI-BF4 wall collision: {case_dir.name}
clear
units real
atom_style full
boundary m m m
read_data initial.data

pair_style reaxff NULL
pair_coeff * * {FFIELD} N C H B F
fix charge all qeq/reaxff 1 0.0 10.0 1.0e-4 reaxff
fix integrate all nve

region wall_region block INF {WALL_X:.8f} INF INF INF INF units box
fix wall all wall/region wall_region lj126 1.0 1.0 2.5

compute tcom all temp/com
thermo_style custom step time c_tcom pe ke etotal atoms
thermo 100
timestep {DT_FS}

dump trajectory all custom {DUMP_INTERVAL} trajectory.lammpstrj id type q x y z vx vy vz
dump_modify trajectory sort id units yes
fix bond_output all reaxff/bonds {DUMP_INTERVAL} bonds.reax

run {steps}
write_data final.data
"""
    (case_dir / "in.case").write_text(input_text, encoding="utf-8")


def parse_dump(path: Path) -> list[dict[str, np.ndarray | int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frames: list[dict[str, np.ndarray | int]] = []
    index = 0
    while index < len(lines):
        if lines[index] != "ITEM: TIMESTEP":
            index += 1
            continue
        step = int(lines[index + 1])
        atom_count = int(lines[index + 3])
        header_index = index + 8
        fields = lines[header_index].split()[2:]
        records = np.array(
            [[float(value) for value in lines[header_index + 1 + row].split()] for row in range(atom_count)]
        )
        order = np.argsort(records[:, fields.index("id")])
        records = records[order]
        frames.append(
            {
                "step": step,
                "ids": records[:, fields.index("id")].astype(int),
                "types": records[:, fields.index("type")].astype(int),
                "positions": records[:, [fields.index("x"), fields.index("y"), fields.index("z")]],
            }
        )
        index = header_index + 1 + atom_count
    return frames


def components_from_distances(positions: np.ndarray, cutoff: float = 2.0) -> list[list[int]]:
    atom_count = len(positions)
    adjacency = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2) < cutoff
    unseen = set(range(atom_count))
    components: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbors = set(np.flatnonzero(adjacency[current])) & unseen
            unseen -= neighbors
            component |= neighbors
            stack.extend(neighbors)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), values[0]))


def component_formula(types: np.ndarray, component: list[int]) -> str:
    counts = Counter(TYPE_SYMBOLS[int(types[index])] for index in component)
    return "".join(
        symbol + (str(counts[symbol]) if counts[symbol] > 1 else "")
        for symbol in FORMULA_ORDER
        if counts[symbol]
    )


def component_mass(types: np.ndarray, component: list[int]) -> float:
    return sum(TYPE_MASSES[int(types[index])] for index in component)


def analyze_case(case_dir: Path, elapsed_seconds: float | None = None) -> dict[str, object]:
    metadata = json.loads((case_dir / "initialization.json").read_text(encoding="utf-8"))
    frames = parse_dump(case_dir / "trajectory.lammpstrj")
    if not frames:
        raise RuntimeError(f"No trajectory frames found for {case_dir.name}")
    final = frames[-1]
    types = np.asarray(final["types"])
    positions = np.asarray(final["positions"])
    components = components_from_distances(positions, cutoff=5.0)
    native_components = components_from_distances(positions, cutoff=2.0)
    formulas = [component_formula(types, component) for component in components]
    masses = [component_mass(types, component) for component in components]
    native_formulas = [component_formula(types, component) for component in native_components]
    native_masses = [component_mass(types, component) for component in native_components]
    persistent = all(
        len(components_from_distances(np.asarray(frame["positions"]), cutoff=5.0)) > 2
        for frame in frames[-5:]
    )

    atom_masses = np.array([TYPE_MASSES[int(value)] for value in types])
    cation_com = np.average(positions[:19], axis=0, weights=atom_masses[:19])
    anion_com = np.average(positions[19:], axis=0, weights=atom_masses[19:])
    ion_separation = float(np.linalg.norm(cation_com - anion_com))

    summary: dict[str, object] = {
        **metadata,
        "status": "complete",
        "completed_steps": int(final["step"]),
        "completed_time_fs": int(final["step"]) * DT_FS,
        "trajectory_frames": len(frames),
        "final_fragment_count": len(components),
        "final_fragment_formulas": formulas,
        "final_fragment_masses_amu": masses,
        "reax_native_2A_fragment_count": len(native_components),
        "reax_native_2A_fragment_formulas": native_formulas,
        "reax_native_2A_fragment_masses_amu": native_masses,
        "covalent_breakup": len(components) > 2,
        "persistent_endpoint_breakup": persistent,
        "final_ion_com_separation_A": ion_separation,
        "ionic_dissociation_10A": ion_separation > 10.0,
        "connectivity_definition": (
            "distance < 5.0 A for direct Laws-Petro DFT comparison; "
            "parallel 2.0 A fields reproduce repository identify_species.m"
        ),
        "elapsed_seconds": elapsed_seconds,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_case(case_dir: Path, resume: bool) -> dict[str, object]:
    summary_path = case_dir / "summary.json"
    if resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        return analyze_case(case_dir, elapsed_seconds=previous.get("elapsed_seconds"))
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    with (case_dir / "screen.log").open("w", encoding="utf-8") as screen:
        result = subprocess.run(
            [str(LAMMPS), "-in", "in.case", "-log", "lammps.log"],
            cwd=case_dir,
            env=environment,
            stdout=screen,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"{case_dir.name} failed; inspect {case_dir / 'screen.log'}")
    return analyze_case(case_dir, elapsed_seconds=elapsed)


def write_campaign_tables(results_dir: Path, summaries: list[dict[str, object]]) -> None:
    summaries = sorted(summaries, key=lambda row: (float(row["energy_eV"]), str(row["orientation"])))
    fields = [
        "case_id",
        "energy_eV",
        "orientation",
        "dft_phi_y_deg",
        "dft_phi_z_deg",
        "initialized_translational_energy_eV",
        "retained_internal_kinetic_energy_eV",
        "completed_steps",
        "completed_time_fs",
        "final_fragment_count",
        "final_fragment_formulas",
        "final_fragment_masses_amu",
        "reax_native_2A_fragment_count",
        "reax_native_2A_fragment_formulas",
        "reax_native_2A_fragment_masses_amu",
        "covalent_breakup",
        "persistent_endpoint_breakup",
        "final_ion_com_separation_A",
        "ionic_dissociation_10A",
        "elapsed_seconds",
        "status",
    ]
    with (results_dir / "case_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {field: summary.get(field) for field in fields}
            row["final_fragment_formulas"] = ";".join(summary["final_fragment_formulas"])
            row["final_fragment_masses_amu"] = ";".join(
                f"{float(value):.6f}" for value in summary["final_fragment_masses_amu"]
            )
            row["reax_native_2A_fragment_formulas"] = ";".join(
                summary["reax_native_2A_fragment_formulas"]
            )
            row["reax_native_2A_fragment_masses_amu"] = ";".join(
                f"{float(value):.6f}" for value in summary["reax_native_2A_fragment_masses_amu"]
            )
            writer.writerow(row)

    spectrum_rows: list[dict[str, object]] = []
    for energy in sorted({float(row["energy_eV"]) for row in summaries}):
        counter: Counter[int] = Counter()
        for summary in summaries:
            if float(summary["energy_eV"]) != energy:
                continue
            counter.update(round(float(value)) for value in summary["final_fragment_masses_amu"])
        total = sum(counter.values())
        for mass, occurrences in sorted(counter.items()):
            spectrum_rows.append(
                {
                    "model": "ReaxFF-reduced42",
                    "energy_eV": energy,
                    "nominal_mass": mass,
                    "occurrences": occurrences,
                    "fragment_occurrence_probability": occurrences / total,
                }
            )
    with (results_dir / "mass_spectrum.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spectrum_rows[0]))
        writer.writeheader()
        writer.writerows(spectrum_rows)

    energy_rows = []
    for energy in sorted({float(row["energy_eV"]) for row in summaries}):
        group = [row for row in summaries if float(row["energy_eV"]) == energy]
        energy_rows.append(
            {
                "energy_eV": energy,
                "n_cases": len(group),
                "covalent_breakup_cases": sum(bool(row["covalent_breakup"]) for row in group),
                "covalent_breakup_fraction": np.mean([bool(row["covalent_breakup"]) for row in group]),
                "ionic_dissociation_cases": sum(bool(row["ionic_dissociation_10A"]) for row in group),
                "ionic_dissociation_fraction": np.mean(
                    [bool(row["ionic_dissociation_10A"]) for row in group]
                ),
                "mean_final_fragment_count": np.mean(
                    [int(row["final_fragment_count"]) for row in group]
                ),
                "mean_elapsed_seconds": np.mean(
                    [float(row["elapsed_seconds"]) for row in group if row["elapsed_seconds"] is not None]
                ),
            }
        )
    with (results_dir / "energy_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(energy_rows[0]))
        writer.writeheader()
        writer.writerows(energy_rows)


def progress(done: int, total: int) -> str:
    width = 30
    filled = round(width * done / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} ({100 * done / total:5.1f}%)"


def main() -> None:
    global LAMMPS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lammps",
        type=Path,
        default=LAMMPS,
        help="LAMMPS executable with the REAXFF package (or set LAMMPS_EXECUTABLE).",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--energies", nargs="+", type=float, default=ENERGIES_EV)
    parser.add_argument("--orientations", nargs="+", choices=ORIENTATIONS, default=ORIENTATIONS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    LAMMPS = args.lammps.expanduser().resolve()
    if not LAMMPS.exists():
        parser.error(f"LAMMPS executable not found: {LAMMPS}")
    if args.steps < 1 or args.jobs < 1:
        parser.error("--steps and --jobs must be positive")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = []
    for energy in args.energies:
        for orientation in args.orientations:
            directory = results_dir / case_id(energy, orientation)
            generate_case(directory, energy, orientation, args.steps)
            case_dirs.append(directory)
    print(f"Generated {len(case_dirs)} cases in {results_dir}", flush=True)
    if args.generate_only:
        return

    summaries: list[dict[str, object]] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_case, directory, args.resume): directory for directory in case_dirs}
        for done, future in enumerate(as_completed(futures), start=1):
            directory = futures[future]
            try:
                summary = future.result()
                summaries.append(summary)
                detail = (
                    f"{directory.name}: {summary['final_fragment_count']} fragments, "
                    f"{float(summary['elapsed_seconds'] or 0):.2f} s"
                )
            except Exception as exc:
                failures.append(f"{directory.name}: {exc}")
                detail = failures[-1]
            print(f"{progress(done, len(case_dirs))} {detail}", flush=True)

    if summaries:
        write_campaign_tables(results_dir, summaries)
    if failures:
        (results_dir / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} cases failed; see {results_dir / 'failures.txt'}")
    print(f"Completed {len(summaries)} cases. Summary: {results_dir / 'case_summary.csv'}")


if __name__ == "__main__":
    main()
