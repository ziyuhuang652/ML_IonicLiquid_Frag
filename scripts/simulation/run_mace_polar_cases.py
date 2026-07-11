#!/usr/bin/env python3
"""
Run an explicit MACE-POLAR-1 collision campaign.

Default cases:
  - EMI-BF4 repulsive-wall collision at 3, 4, 5, 6 eV
  - N2 impact on EMI at 3, 4, 5, 6 eV
  - N2 impact on BF4 at 3, 4, 5, 6 eV

Energy is the total translational kinetic energy assigned to the moving
subsystem. Energies, case groups, and the results directory are configurable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.data import covalent_radii
from ase.io import Trajectory, read, write
from ase.md.verlet import VelocityVerlet


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
RESULTS_DIR = ROOT / "results"
ENERGIES_EV = [3, 4, 5, 6]
DT_FS = 0.5
WALL_STEPS = 2000
N2_STEPS = 800
SAVE_INTERVAL = 10
PERSISTENCE_SAMPLES = 5
RNG_SEED = 20260610
MACE_EVAL_CELL_A = 50.0
CLUSTER_INPUT_DIR = ROOT.parent / "References" / "extracted_structures"
N2_TARGETS = {
    "EMI": {
        "path": INPUT_DIR / "EMI.dat",
        "format": "lammps-data",
        "charge": 1,
        "orientation": "emi",
    },
    "BF4": {
        "path": INPUT_DIR / "BF4.dat",
        "format": "lammps-data",
        "charge": -1,
        "orientation": "bf4",
    },
    "EMIBF4EMI": {
        "path": CLUSTER_INPUT_DIR / "EMIM_BF4_EMIM_plus.xyz",
        "format": "xyz",
        "charge": 1,
        "orientation": "principal",
    },
    "EMIBF4BF4": {
        "path": CLUSTER_INPUT_DIR / "EMIM_BF4_BF4_minus.xyz",
        "format": "xyz",
        "charge": -1,
        "orientation": "principal",
    },
}


def get_calculator(
    model: str,
    device: str,
    dtype: str,
    backend: str = "mace-polar",
    modal: str | None = None,
    enable_flash: bool = False,
):
    if backend == "mace-polar":
        from mace.calculators import mace_polar

        return mace_polar(model=model, device=device, default_dtype=dtype)
    if backend == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        kwargs = {"model": model, "device": device, "enable_flash": enable_flash}
        if modal:
            kwargs["modal"] = modal
        return SevenNetCalculator(**kwargs)
    raise ValueError(f"Unknown calculator backend: {backend}")


def centered_mace_atoms(atoms):
    """Return a compact nonperiodic copy for MACE-POLAR long-range evaluation."""
    centered_atoms = atoms.copy()
    positions = centered_atoms.get_positions()
    midpoint = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    centered_atoms.set_positions(positions - midpoint)
    centered_atoms.set_cell(np.eye(3) * MACE_EVAL_CELL_A)
    centered_atoms.set_pbc([False, False, False])
    return centered_atoms


class TerminalGeometryReached(RuntimeError):
    def __init__(self, reason, maximum_separation_A, maximum_radius_A):
        super().__init__(reason)
        self.reason = reason
        self.maximum_separation_A = maximum_separation_A
        self.maximum_radius_A = maximum_radius_A


def raise_if_terminal_geometry(atoms, max_extent_A=None, terminal_sphere_radius_A=None):
    maximum_separation, maximum_radius = geometry_extent_metrics(atoms)
    if terminal_sphere_radius_A is not None and maximum_radius >= terminal_sphere_radius_A:
        raise TerminalGeometryReached(
            f"terminal_sphere_radius_reached_{terminal_sphere_radius_A:g}_A",
            maximum_separation,
            maximum_radius,
        )
    if max_extent_A is not None and maximum_separation >= max_extent_A:
        raise TerminalGeometryReached(
            f"terminal_fragment_separation_reached_{max_extent_A:g}_A",
            maximum_separation,
            maximum_radius,
        )


class RepulsiveWallCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calculator,
        wall_position=0.0,
        wall_stiffness=100.0,
        max_extent_A=None,
        terminal_sphere_radius_A=None,
    ):
        super().__init__()
        self.base_calc = base_calculator
        self.wall_position = wall_position
        self.wall_stiffness = wall_stiffness
        self.max_extent_A = max_extent_A
        self.terminal_sphere_radius_A = terminal_sphere_radius_A

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        raise_if_terminal_geometry(atoms, self.max_extent_A, self.terminal_sphere_radius_A)
        self.base_calc.calculate(centered_mace_atoms(atoms), properties, all_changes)

        base_energy = self.base_calc.results.get("energy", 0.0)
        base_forces = self.base_calc.results.get("forces", np.zeros((len(atoms), 3)))
        positions = atoms.get_positions()

        wall_energy = 0.0
        wall_forces = np.zeros_like(base_forces)
        below_wall = positions[:, 2] < self.wall_position
        if np.any(below_wall):
            penetration = self.wall_position - positions[below_wall, 2]
            wall_energy = 0.5 * self.wall_stiffness * np.sum(penetration**2)
            wall_forces[below_wall, 2] = self.wall_stiffness * penetration

        self.results = {
            "energy": base_energy + wall_energy,
            "forces": base_forces + wall_forces,
        }


class CenteredCalculator(Calculator):
    """Evaluate a translationally invariant isolated system near the origin."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calculator, max_extent_A=None, terminal_sphere_radius_A=None):
        super().__init__()
        self.base_calc = base_calculator
        self.max_extent_A = max_extent_A
        self.terminal_sphere_radius_A = terminal_sphere_radius_A

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        raise_if_terminal_geometry(atoms, self.max_extent_A, self.terminal_sphere_radius_A)
        self.base_calc.calculate(centered_mace_atoms(atoms), properties, all_changes)
        self.results = {
            "energy": self.base_calc.results.get("energy", 0.0),
            "forces": self.base_calc.results.get("forces", np.zeros((len(atoms), 3))),
        }


class DetachableN2Calculator(Calculator):
    """Drop negligible target-N2 coupling after the outgoing collision partners separate."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calculator,
        target_indices,
        n2_indices,
        detach_distance_A,
        target_fragment_detach_distance_A=None,
        max_extent_A=None,
        terminal_sphere_radius_A=None,
    ):
        super().__init__()
        self.base_calc = base_calculator
        self.target_indices = np.asarray(target_indices, dtype=int)
        self.n2_indices = np.asarray(n2_indices, dtype=int)
        self.detach_distance_A = detach_distance_A
        self.target_fragment_detach_distance_A = target_fragment_detach_distance_A
        self.max_extent_A = max_extent_A
        self.terminal_sphere_radius_A = terminal_sphere_radius_A
        self.initial_distance_A = None
        self.minimum_distance_A = math.inf
        self.detached = False
        self.detach_distance_observed_A = None
        self.force_calls = 0
        self.detach_force_call = None
        self.detached_n2_reference_energy = None
        self.target_fragments_decoupled = False
        self.target_fragment_detach_force_call = None
        self.target_fragment_charge_assignment = None

    def _minimum_target_n2_distance(self, atoms):
        positions = atoms.get_positions()
        displacement = (
            positions[self.target_indices][:, None, :]
            - positions[self.n2_indices][None, :, :]
        )
        return float(np.linalg.norm(displacement, axis=-1).min())

    def _evaluate_subset(self, atoms, indices, charge, properties):
        subset = atoms[indices]
        subset.info.update(atoms.info)
        subset.info["charge"] = charge
        self.base_calc.calculate(centered_mace_atoms(subset), properties, all_changes)
        energy = self.base_calc.results.get("energy", 0.0)
        forces = self.base_calc.results.get("forces", np.zeros((len(subset), 3))).copy()
        return energy, forces

    def _target_subsystems(self, atoms):
        target_atoms = atoms[self.target_indices]
        covalent_groups = fragments(target_atoms)
        if (
            self.target_fragment_detach_distance_A is None
            or len(covalent_groups) <= 1
        ):
            return [self.target_indices.tolist()]

        positions = target_atoms.get_positions()
        adjacency = {index: [] for index in range(len(covalent_groups))}
        for left in range(len(covalent_groups)):
            for right in range(left + 1, len(covalent_groups)):
                displacement = (
                    positions[covalent_groups[left]][:, None, :]
                    - positions[covalent_groups[right]][None, :, :]
                )
                if (
                    np.linalg.norm(displacement, axis=-1).min()
                    < self.target_fragment_detach_distance_A
                ):
                    adjacency[left].append(right)
                    adjacency[right].append(left)

        components = []
        seen = set()
        for start in adjacency:
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            component = []
            while stack:
                group_index = stack.pop()
                component.extend(covalent_groups[group_index])
                for neighbor in adjacency[group_index]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component))
        return [
            self.target_indices[np.asarray(component, dtype=int)].tolist()
            for component in components
        ]

    def _assign_subsystem_charges(self, atoms, subsystems):
        charges = [0] * len(subsystems)
        total_charge = int(atoms.info["charge"])
        if total_charge == 0 or len(subsystems) == 1:
            charges[0] = total_charge
            return charges

        symbols = atoms.get_chemical_symbols()
        if total_charge > 0:
            charged_index = max(
                range(len(subsystems)),
                key=lambda index: (
                    sum(symbols[atom] == "N" for atom in subsystems[index]),
                    len(subsystems[index]),
                ),
            )
        else:
            fluorine_fragments = [
                index
                for index, subsystem in enumerate(subsystems)
                if any(symbols[atom] == "F" for atom in subsystem)
            ]
            candidates = fluorine_fragments or list(range(len(subsystems)))
            charged_index = min(candidates, key=lambda index: len(subsystems[index]))
        charges[charged_index] = total_charge
        return charges

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.force_calls += 1
        distance = self._minimum_target_n2_distance(atoms)
        if self.initial_distance_A is None:
            self.initial_distance_A = distance
        self.minimum_distance_A = min(self.minimum_distance_A, distance)
        approached = self.minimum_distance_A < self.initial_distance_A - 0.1
        if not self.detached and approached and distance >= self.detach_distance_A:
            self.detached = True
            self.detach_distance_observed_A = distance
            self.detach_force_call = self.force_calls
            velocities = atoms.get_velocities()
            n2_masses = atoms.get_masses()[self.n2_indices]
            n2_com_velocity = np.average(
                velocities[self.n2_indices], axis=0, weights=n2_masses
            )
            velocities[self.n2_indices] = n2_com_velocity
            atoms.set_velocities(velocities)

        if not self.detached:
            raise_if_terminal_geometry(atoms, self.max_extent_A, self.terminal_sphere_radius_A)
            self.base_calc.calculate(centered_mace_atoms(atoms), properties, all_changes)
            self.results = {
                "energy": self.base_calc.results.get("energy", 0.0),
                "forces": self.base_calc.results.get("forces", np.zeros((len(atoms), 3))),
            }
            return

        subsystems = self._target_subsystems(atoms)
        subsystem_charges = self._assign_subsystem_charges(atoms, subsystems)
        if len(subsystems) > 1:
            self.target_fragments_decoupled = True
            if self.target_fragment_detach_force_call is None:
                self.target_fragment_detach_force_call = self.force_calls
            self.target_fragment_charge_assignment = [
                {
                    "indices": [int(index) for index in subsystem],
                    "charge": charge,
                    "formula": formula_for(atoms, subsystem),
                }
                for subsystem, charge in zip(subsystems, subsystem_charges)
            ]

        if self.detached_n2_reference_energy is None:
            n2_energy, _ = self._evaluate_subset(atoms, self.n2_indices, 0, properties)
            self.detached_n2_reference_energy = float(n2_energy)
        forces = np.zeros((len(atoms), 3))
        target_energy = 0.0
        for subsystem, charge in zip(subsystems, subsystem_charges):
            subsystem_energy, subsystem_forces = self._evaluate_subset(
                atoms, subsystem, charge, properties
            )
            target_energy += subsystem_energy
            forces[subsystem] = subsystem_forces
        self.results = {
            "energy": target_energy + self.detached_n2_reference_energy,
            "forces": forces,
        }


def bonds_from_connectivity(atoms, cutoff_multiplier=1.3, ignored_symbols=frozenset()):
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    bonds = []
    for i in range(len(atoms)):
        if symbols[i] in ignored_symbols:
            continue
        for j in range(i + 1, len(atoms)):
            if symbols[j] in ignored_symbols:
                continue
            cutoff = (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number]) * cutoff_multiplier
            if np.linalg.norm(positions[i] - positions[j]) < cutoff:
                bonds.append((i, j, symbols[i], symbols[j]))
    return bonds


def bond_distances(atoms, bonds):
    positions = atoms.get_positions()
    return np.array([np.linalg.norm(positions[i] - positions[j]) for i, j, _, _ in bonds])


def geometry_extent_metrics(atoms):
    if len(atoms) <= 1:
        return 0.0, 0.0
    positions = atoms.get_positions()
    displacement = positions[:, None, :] - positions[None, :, :]
    maximum_separation = float(np.linalg.norm(displacement, axis=-1).max())
    center = positions.mean(axis=0)
    maximum_radius = float(np.linalg.norm(positions - center, axis=1).max())
    return maximum_separation, maximum_radius


def fragments(atoms, cutoff_multiplier=1.3, ignored_symbols=frozenset()):
    symbols = atoms.get_chemical_symbols()
    active = [i for i, sym in enumerate(symbols) if sym not in ignored_symbols]
    active_set = set(active)
    adjacency = {i: [] for i in active}
    positions = atoms.get_positions()

    for pos, i in enumerate(active):
        for j in active[pos + 1 :]:
            cutoff = (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number]) * cutoff_multiplier
            if np.linalg.norm(positions[i] - positions[j]) < cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)

    seen = set()
    out = []
    for start in active:
        if start in seen:
            continue
        stack = [start]
        group = []
        seen.add(start)
        while stack:
            node = stack.pop()
            group.append(node)
            for nbr in adjacency[node]:
                if nbr in active_set and nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        out.append(sorted(group))
    return sorted(out, key=lambda x: (-len(x), x[0]))


def formula_for(atoms, indices):
    counts = Counter(atoms[i].symbol for i in indices)
    return "".join(elem if counts[elem] == 1 else f"{elem}{counts[elem]}" for elem in sorted(counts))


def set_downward_energy(atoms, indices, energy_ev):
    mass = sum(atoms[i].mass for i in indices)
    velocity = math.sqrt(2.0 * energy_ev / mass)
    velocities = np.zeros((len(atoms), 3))
    for idx in indices:
        velocities[idx, 2] = -velocity
    atoms.set_velocities(velocities)
    initialized_energy = float(atoms.get_kinetic_energy())
    if not math.isclose(initialized_energy, energy_ev, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(
            f"Impact-energy initialization failed: requested {energy_ev} eV, "
            f"initialized {initialized_energy} eV"
        )
    return {
        "impact_velocity_ase": velocity,
        "impact_velocity_A_per_fs": velocity * units.fs,
        "requested_impact_energy_eV": float(energy_ev),
        "initialized_kinetic_energy_eV": initialized_energy,
    }


def rotate_vector_to(atoms, source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if np.linalg.norm(source) < 1e-12:
        raise ValueError("Cannot orient a near-zero source vector")
    atoms.rotate(source, target, center="COM")


def wall_orientation_names(scheme):
    if scheme == "single":
        return ["reference"]
    if scheme == "paper6":
        return ["dipole_px", "dipole_nx", "dipole_py", "dipole_ny", "dipole_pz", "dipole_nz"]
    raise ValueError(f"Unknown wall orientation scheme: {scheme}")


def apply_wall_orientation(atoms, orientation):
    if orientation == "reference":
        return
    charges = atoms.arrays.get("initial_charges", atoms.arrays.get("mmcharges"))
    if charges is None:
        raise ValueError("Wall orientation requires input partial charges to define the molecular dipole")
    origin = atoms.get_center_of_mass()
    dipole = np.sum(charges[:, None] * (atoms.get_positions() - origin), axis=0)
    targets = {
        "dipole_px": [1, 0, 0],
        "dipole_nx": [-1, 0, 0],
        "dipole_py": [0, 1, 0],
        "dipole_ny": [0, -1, 0],
        "dipole_pz": [0, 0, 1],
        "dipole_nz": [0, 0, -1],
    }
    rotate_vector_to(atoms, dipole, targets[orientation])


def n2_target_orientation_names(target_name, scheme):
    if scheme == "single":
        return ["reference"]
    if scheme == "pilot3":
        orientation_type = N2_TARGETS[target_name]["orientation"]
        if orientation_type in {"emi", "principal"}:
            return ["principal_1", "principal_2", "principal_3"]
        return ["vertex", "edge", "face"]
    raise ValueError(f"Unknown N2 orientation scheme: {scheme}")


def apply_n2_target_orientation(atoms, target_name, orientation):
    if orientation == "reference":
        return
    positions = atoms.get_positions()
    if N2_TARGETS[target_name]["orientation"] in {"emi", "principal"}:
        masses = atoms.get_masses()
        centered = positions - atoms.get_center_of_mass()
        inertia = np.zeros((3, 3))
        for mass, vector in zip(masses, centered):
            inertia += mass * (np.dot(vector, vector) * np.eye(3) - np.outer(vector, vector))
        _, axes = np.linalg.eigh(inertia)
        axis_index = int(orientation.rsplit("_", 1)[1]) - 1
        rotate_vector_to(atoms, axes[:, axis_index], [0, 0, 1])
        return

    symbols = atoms.get_chemical_symbols()
    b_index = symbols.index("B")
    f_indices = [i for i, symbol in enumerate(symbols) if symbol == "F"]
    b_position = positions[b_index]
    if orientation == "vertex":
        vector = positions[f_indices[0]] - b_position
    elif orientation == "edge":
        vector = positions[f_indices[:2]].mean(axis=0) - b_position
    elif orientation == "face":
        vector = positions[f_indices[:3]].mean(axis=0) - b_position
    else:
        raise ValueError(f"Unknown BF4 orientation: {orientation}")
    rotate_vector_to(atoms, vector, [0, 0, 1])


def setup_wall_case(energy_ev, orientation="reference"):
    atoms = read(INPUT_DIR / "EMIBF4.dat", format="lammps-data", style="full")
    atoms.set_pbc([False, False, False])
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    apply_wall_orientation(atoms, orientation)
    positions = atoms.get_positions()
    positions[:, 2] -= positions[:, 2].mean() - 10.0
    atoms.set_positions(positions)
    impact_indices = list(range(len(atoms)))
    impact = set_downward_energy(atoms, impact_indices, energy_ev)
    return atoms, {**impact, "impact_indices": impact_indices, "orientation": orientation}


def setup_n2_case(
    target_name,
    energy_ev,
    rng,
    moving_species="n2",
    target_orientation="reference",
    n2_orientation="random",
):
    target_spec = N2_TARGETS[target_name]
    read_kwargs = {"format": target_spec["format"]}
    if target_spec["format"] == "lammps-data":
        read_kwargs["style"] = "full"
    atoms = read(target_spec["path"], **read_kwargs)
    atoms.set_pbc([False, False, False])
    apply_n2_target_orientation(atoms, target_name, target_orientation)
    target_indices = list(range(len(atoms)))
    n2 = molecule("N2")
    if n2_orientation == "random":
        n2.rotate(float(rng.uniform(0, 360)), "x")
        n2.rotate(float(rng.uniform(0, 360)), "y")
    elif n2_orientation == "perpendicular":
        n2.rotate(90.0, "x")
    else:
        raise ValueError(f"Unknown N2 orientation: {n2_orientation}")

    direction = np.array([0.0, 0.0, 1.0])
    target_com = atoms.get_center_of_mass()
    if target_spec["orientation"] == "principal":
        n2_center = target_com.copy()
        n2_center[2] = atoms.get_positions()[:, 2].max() + 3.0
    else:
        n2_center = target_com + direction * 5.0
    n2.translate(n2_center - n2.get_center_of_mass())
    atoms += n2
    atoms.info["charge"] = target_spec["charge"]
    atoms.info["spin"] = 1
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    n2_indices = [len(atoms) - 2, len(atoms) - 1]
    if moving_species == "n2":
        impact_indices = n2_indices
        impact_direction = -direction
    elif moving_species == "target":
        impact_indices = target_indices
        impact_direction = direction
    else:
        raise ValueError(f"Unknown moving species: {moving_species}")

    mass = sum(atoms[i].mass for i in impact_indices)
    velocity = math.sqrt(2.0 * energy_ev / mass)
    velocities = np.zeros((len(atoms), 3))
    for idx in impact_indices:
        velocities[idx] = impact_direction * velocity
    atoms.set_velocities(velocities)
    initialized_energy = float(atoms.get_kinetic_energy())
    if not math.isclose(initialized_energy, energy_ev, rel_tol=1e-10, abs_tol=1e-10):
        raise RuntimeError(
            f"Impact-energy initialization failed: requested {energy_ev} eV, "
            f"initialized {initialized_energy} eV"
        )
    impact = {
        "impact_velocity_ase": velocity,
        "impact_velocity_A_per_fs": velocity * units.fs,
        "requested_impact_energy_eV": float(energy_ev),
        "initialized_kinetic_energy_eV": initialized_energy,
    }
    return atoms, {
        **impact,
        "moving_species": moving_species,
        "impact_indices": impact_indices,
        "n2_indices": n2_indices,
        "target_indices": target_indices,
        "direction": direction.tolist(),
        "initial_target_n2_com_distance_A": float(np.linalg.norm(n2_center - target_com)),
        "target_orientation": target_orientation,
        "n2_orientation": n2_orientation,
    }


def run_md_case(
    case,
    atoms,
    calc,
    nsteps,
    dt_fs=DT_FS,
    ignored_fragment_symbols=frozenset(),
    max_extent_A=None,
    terminal_sphere_radius_A=None,
):
    case_dir = RESULTS_DIR / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    initial_atoms = atoms.copy()
    initial_bonds = bonds_from_connectivity(initial_atoms, ignored_symbols=ignored_fragment_symbols)
    initial_distances = bond_distances(initial_atoms, initial_bonds)
    target_indices = case.get("target_indices", list(range(len(atoms))))
    initial_target_atoms = initial_atoms[target_indices]
    initial_target_bonds = bonds_from_connectivity(initial_target_atoms)
    initial_target_distances = bond_distances(initial_target_atoms, initial_target_bonds)

    atoms.calc = case["calculator_wrapper"](calc) if case.get("calculator_wrapper") else calc
    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)

    traj_path = case_dir / f"{case['case_id']}.traj"
    xyz_path = case_dir / f"{case['case_id']}.xyz"
    energy_trace = []
    fragment_trace = []
    bond_trace = []
    target_fragment_trace = []
    target_bond_trace = []

    traj = Trajectory(str(traj_path), "w", atoms)
    write(str(xyz_path), atoms, append=False, format="xyz")

    t0 = time.time()
    termination_reason = "requested_steps_completed"
    calculator_error = None
    completed_steps = 0
    for step in range(nsteps + 1):
        completed_steps = step
        if terminal_sphere_radius_A is not None and len(atoms) > 1:
            _, maximum_radius = geometry_extent_metrics(atoms)
            if maximum_radius >= terminal_sphere_radius_A:
                termination_reason = f"terminal_sphere_radius_reached_{terminal_sphere_radius_A:g}_A"
                print(
                    f"Stopping {case['case_id']} at step {step}: "
                    f"maximum radius {maximum_radius:.2f} A from geometric center",
                    flush=True,
                )
                break
        if max_extent_A is not None and len(atoms) > 1:
            maximum_separation, _ = geometry_extent_metrics(atoms)
            if maximum_separation >= max_extent_A:
                termination_reason = f"terminal_fragment_separation_reached_{max_extent_A:g}_A"
                print(
                    f"Stopping {case['case_id']} at step {step}: "
                    f"maximum separation {maximum_separation:.2f} A",
                    flush=True,
                )
                break
        if step % SAVE_INTERVAL == 0:
            traj.write(atoms)
            if step > 0:
                write(str(xyz_path), atoms, append=True, format="xyz")
            current_fragments = fragments(atoms, ignored_symbols=ignored_fragment_symbols)
            fragment_trace.append(
                {
                    "step": step,
                    "n_fragments": len(current_fragments),
                    "fragment_sizes": [len(f) for f in current_fragments],
                    "fragment_formulas": [formula_for(atoms, f) for f in current_fragments],
                }
            )
            bond_trace.append(bond_distances(atoms, initial_bonds).tolist())
            target_atoms = atoms[target_indices]
            current_target_fragments = fragments(target_atoms)
            target_fragment_trace.append(
                {
                    "step": step,
                    "n_fragments": len(current_target_fragments),
                    "fragment_sizes": [len(f) for f in current_target_fragments],
                    "fragment_formulas": [
                        formula_for(target_atoms, f) for f in current_target_fragments
                    ],
                }
            )
            target_bond_trace.append(
                bond_distances(target_atoms, initial_target_bonds).tolist()
            )
            energy_trace.append(
                {
                    "step": step,
                    "potential_eV": float(atoms.get_potential_energy()),
                    "kinetic_eV": float(atoms.get_kinetic_energy()),
                    "total_eV": float(atoms.get_potential_energy() + atoms.get_kinetic_energy()),
                }
            )
        if step == nsteps:
            break
        try:
            dyn.run(1)
        except TerminalGeometryReached as err:
            termination_reason = err.reason
            print(
                f"Stopping {case['case_id']} at step {step}: "
                f"terminal geometry reached inside force evaluation "
                f"(separation {err.maximum_separation_A:.2f} A, radius {err.maximum_radius_A:.2f} A)",
                flush=True,
            )
            break
        except Exception as err:
            message = str(err)
            longrange_failure = any(
                marker in message
                for marker in (
                    "graph_longrange",
                    "compute_k_vectors",
                    "cartesian_prod",
                    "integer multiplication overflow",
                    "CUDA error: out of memory",
                    "CUDA out of memory",
                )
            )
            if not longrange_failure:
                raise
            maximum_separation, maximum_radius = geometry_extent_metrics(atoms)
            termination_reason = "terminal_mace_polar_longrange_grid_failed"
            calculator_error = {
                "type": type(err).__name__,
                "message": message.splitlines()[0],
                "maximum_separation_A": maximum_separation,
                "maximum_radius_A": maximum_radius,
            }
            print(
                f"Stopping {case['case_id']} at step {step}: "
                f"MACE-POLAR long-range grid failed after geometry expansion "
                f"(separation {maximum_separation:.2f} A, radius {maximum_radius:.2f} A)",
                flush=True,
            )
            break

    traj.close()
    elapsed = time.time() - t0

    final_fragments = fragments(atoms, ignored_symbols=ignored_fragment_symbols)
    final_distances = bond_distances(atoms, initial_bonds)
    final_maximum_separation_A, final_maximum_radius_A = geometry_extent_metrics(atoms)
    broken = []
    for idx, (bond, d0, d1) in enumerate(zip(initial_bonds, initial_distances, final_distances)):
        if d1 > 1.5 * d0:
            i, j, si, sj = bond
            broken.append(
                {
                    "bond_index": idx,
                    "atoms": [int(i), int(j)],
                    "symbols": f"{si}-{sj}",
                    "initial_distance_A": float(d0),
                    "final_distance_A": float(d1),
                    "elongation_ratio": float(d1 / d0),
                }
            )

    bond_trace_array = np.array(bond_trace)
    target_bond_trace_array = np.array(target_bond_trace)
    persistent_broken = []
    first_persistent_break_step = None
    if initial_target_bonds and len(target_bond_trace_array) >= PERSISTENCE_SAMPLES:
        broken_samples = target_bond_trace_array > (
            1.5 * initial_target_distances[None, :]
        )
        for bond_index, bond in enumerate(initial_target_bonds):
            series = broken_samples[:, bond_index]
            first_sample = None
            for sample in range(len(series) - PERSISTENCE_SAMPLES + 1):
                if series[sample : sample + PERSISTENCE_SAMPLES].all():
                    first_sample = sample
                    break
            if first_sample is None:
                continue
            i, j, si, sj = bond
            break_step = first_sample * SAVE_INTERVAL
            persistent_broken.append(
                {
                    "bond_index": bond_index,
                    "atoms": [int(i), int(j)],
                    "symbols": f"{si}-{sj}",
                    "first_persistent_break_step": break_step,
                    "first_persistent_break_time_fs": break_step * dt_fs,
                }
            )
            if first_persistent_break_step is None:
                first_persistent_break_step = break_step
            else:
                first_persistent_break_step = min(first_persistent_break_step, break_step)

    initial_fragment_count = (
        target_fragment_trace[0]["n_fragments"]
        if target_fragment_trace
        else len(fragments(initial_target_atoms))
    )
    fragment_counts = [sample["n_fragments"] for sample in target_fragment_trace]
    first_persistent_fragment_step = None
    for sample in range(len(fragment_counts) - PERSISTENCE_SAMPLES + 1):
        if all(
            count > initial_fragment_count
            for count in fragment_counts[sample : sample + PERSISTENCE_SAMPLES]
        ):
            first_persistent_fragment_step = target_fragment_trace[sample]["step"]
            break

    final_target_atoms = atoms[target_indices]
    final_target_fragments = fragments(final_target_atoms)

    result = {
        **case,
        "success": True,
        "nsteps": nsteps,
        "completed_steps": completed_steps,
        "dt_fs": dt_fs,
        "save_interval": SAVE_INTERVAL,
        "termination_reason": termination_reason,
        "calculator_error": calculator_error,
        "max_extent_A": max_extent_A,
        "terminal_sphere_radius_A": terminal_sphere_radius_A,
        "mace_eval_cell_A": MACE_EVAL_CELL_A,
        "n2_detached_after_collision": bool(getattr(atoms.calc, "detached", False)),
        "n2_detach_distance_observed_A": getattr(
            atoms.calc, "detach_distance_observed_A", None
        ),
        "n2_detach_force_call": getattr(atoms.calc, "detach_force_call", None),
        "minimum_target_n2_distance_A": (
            float(atoms.calc.minimum_distance_A)
            if hasattr(atoms.calc, "minimum_distance_A")
            else None
        ),
        "target_fragments_decoupled": bool(
            getattr(atoms.calc, "target_fragments_decoupled", False)
        ),
        "target_fragment_detach_force_call": getattr(
            atoms.calc, "target_fragment_detach_force_call", None
        ),
        "target_fragment_charge_assignment": getattr(
            atoms.calc, "target_fragment_charge_assignment", None
        ),
        "final_maximum_separation_A": final_maximum_separation_A,
        "final_maximum_radius_A": final_maximum_radius_A,
        "elapsed_time_sec": elapsed,
        "initial_bonds": len(initial_bonds),
        "broken_bonds_count": len(broken),
        "broken_bonds": broken,
        "persistent_broken_bonds_count": len(persistent_broken),
        "persistent_broken_bonds": persistent_broken,
        "first_persistent_bond_break_step": first_persistent_break_step,
        "first_persistent_bond_break_time_fs": (
            first_persistent_break_step * dt_fs
            if first_persistent_break_step is not None
            else None
        ),
        "final_fragments": len(final_fragments),
        "initial_target_fragments": initial_fragment_count,
        "maximum_target_fragments": max(
            fragment_counts, default=len(final_target_fragments)
        ),
        "final_target_fragments": len(final_target_fragments),
        "final_target_fragment_formulas": [
            formula_for(final_target_atoms, f) for f in final_target_fragments
        ],
        "first_persistent_fragmentation_step": first_persistent_fragment_step,
        "first_persistent_fragmentation_time_fs": (
            first_persistent_fragment_step * dt_fs
            if first_persistent_fragment_step is not None
            else None
        ),
        "fragment_formulas": [formula_for(atoms, f) for f in final_fragments],
        "fragment_sizes": [len(f) for f in final_fragments],
        "fragment_trace": fragment_trace,
        "target_fragment_trace": target_fragment_trace,
        "energy_trace": energy_trace,
        "outputs": {
            "trajectory": str(traj_path),
            "xyz": str(xyz_path),
            "summary": str(case_dir / f"{case['case_id']}_summary.json"),
        },
    }
    result.pop("calculator_wrapper", None)

    with open(case_dir / f"{case['case_id']}_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(case_dir / f"{case['case_id']}_bond_trace.npy", bond_trace_array)
    np.save(
        case_dir / f"{case['case_id']}_target_bond_trace.npy",
        target_bond_trace_array,
    )
    write(str(case_dir / f"{case['case_id']}_final.xyz"), atoms)
    return result


def show_progress(completed, total, label="progress"):
    width = 30
    fraction = completed / total if total else 1.0
    filled = min(width, int(round(width * fraction)))
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {completed}/{total} ({100.0 * fraction:5.1f}%) {label}", flush=True)


def main():
    global RESULTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calculator-backend",
        choices=["mace-polar", "sevennet"],
        default="mace-polar",
    )
    parser.add_argument("--model", default="polar-1-m", help="MACE-POLAR model name, e.g. polar-1-m or polar-1-l")
    parser.add_argument(
        "--model-modal",
        default=None,
        help="Optional multi-task model modality, e.g. omol25_low for SevenNet-Omni",
    )
    parser.add_argument(
        "--sevennet-enable-flash",
        action="store_true",
        help="Enable the FlashTP backend in SevenNetCalculator",
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dt-fs",
        type=float,
        default=DT_FS,
        help=f"Fixed MD integration timestep in femtoseconds (default: {DT_FS})",
    )
    parser.add_argument("--energies", nargs="+", type=float, default=ENERGIES_EV)
    parser.add_argument(
        "--case-groups",
        nargs="+",
        choices=["wall", "n2-emi", "n2-bf4", "n2-cluster-cation", "n2-cluster-anion"],
        default=["wall", "n2-emi", "n2-bf4"],
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--n2-moving-species",
        choices=["n2", "target"],
        default="n2",
        help="'target' matches the reference CID paper's ion-projectile laboratory frame",
    )
    parser.add_argument(
        "--wall-orientation-scheme",
        choices=["single", "paper6"],
        default="single",
    )
    parser.add_argument(
        "--n2-orientation-scheme",
        choices=["single", "pilot3"],
        default="single",
    )
    parser.add_argument("--resume", action="store_true", help="Skip cases with an existing summary JSON")
    parser.add_argument(
        "--n2-steps",
        type=int,
        default=N2_STEPS,
        help=f"N2 collision integration steps (default: {N2_STEPS})",
    )
    parser.add_argument(
        "--wall-steps",
        type=int,
        default=WALL_STEPS,
        help=f"Wall collision integration steps (default: {WALL_STEPS})",
    )
    parser.add_argument(
        "--max-extent-A",
        type=float,
        default=40.0,
        help="Stop after terminal fragment separation reaches this extent; use 0 to disable",
    )
    parser.add_argument(
        "--terminal-sphere-radius-A",
        type=float,
        default=0.0,
        help="Stop when any atom is this far from the instantaneous geometric center; use 0 to disable",
    )
    parser.add_argument(
        "--n2-detach-distance-A",
        type=float,
        default=0.0,
        help=(
            "After a collision, evaluate the target and outgoing N2 independently once their "
            "minimum distance reaches this value; use 0 to disable"
        ),
    )
    parser.add_argument(
        "--target-fragment-detach-distance-A",
        type=float,
        default=0.0,
        help=(
            "After N2 departure, evaluate covalent product groups independently once no "
            "intergroup atom pair is within this distance; use 0 to disable"
        ),
    )
    parser.add_argument("--shard-count", type=int, default=1, help="Split the ordered case matrix into this many shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard assigned to this process")
    parser.add_argument("--dry-run", action="store_true", help="Print the case matrix without loading MACE")
    parser.add_argument("--quick", action="store_true", help="Run fewer steps for smoke testing")
    args = parser.parse_args()
    if args.n2_steps < 1 or args.wall_steps < 1:
        parser.error("--n2-steps and --wall-steps must be positive")
    if args.dt_fs <= 0:
        parser.error("--dt-fs must be positive")
    max_extent_A = args.max_extent_A if args.max_extent_A > 0 else None
    terminal_sphere_radius_A = (
        args.terminal_sphere_radius_A if args.terminal_sphere_radius_A > 0 else None
    )
    n2_detach_distance_A = (
        args.n2_detach_distance_A if args.n2_detach_distance_A > 0 else None
    )
    target_fragment_detach_distance_A = (
        args.target_fragment_detach_distance_A
        if args.target_fragment_detach_distance_A > 0
        else None
    )

    RESULTS_DIR = args.results_dir.expanduser().resolve()
    energies = list(args.energies)
    planned_cases = []
    wall_orientations = wall_orientation_names(args.wall_orientation_scheme)
    if "wall" in args.case_groups:
        planned_cases.extend(("wall", "EMIBF4", energy, orientation) for energy in energies for orientation in wall_orientations)
    if "n2-emi" in args.case_groups:
        planned_cases.extend(
            ("n2", "EMI", energy, orientation)
            for energy in energies
            for orientation in n2_target_orientation_names("EMI", args.n2_orientation_scheme)
        )
    if "n2-bf4" in args.case_groups:
        planned_cases.extend(
            ("n2", "BF4", energy, orientation)
            for energy in energies
            for orientation in n2_target_orientation_names("BF4", args.n2_orientation_scheme)
        )
    if "n2-cluster-cation" in args.case_groups:
        planned_cases.extend(
            ("n2", "EMIBF4EMI", energy, orientation)
            for energy in energies
            for orientation in n2_target_orientation_names("EMIBF4EMI", args.n2_orientation_scheme)
        )
    if "n2-cluster-anion" in args.case_groups:
        planned_cases.extend(
            ("n2", "EMIBF4BF4", energy, orientation)
            for energy in energies
            for orientation in n2_target_orientation_names("EMIBF4BF4", args.n2_orientation_scheme)
        )
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")
    assigned_cases = [
        case for index, case in enumerate(planned_cases) if index % args.shard_count == args.shard_index
    ]
    assigned_case_set = set(assigned_cases)

    if args.dry_run:
        print(f"Results directory: {RESULTS_DIR}")
        print(
            f"Backend: {args.calculator_backend}; model: {args.model}; "
            f"modal: {args.model_modal}; device: {args.device}; dtype: {args.dtype}"
        )
        print(f"N2 collision moving species: {args.n2_moving_species}")
        print(f"Wall orientation scheme: {args.wall_orientation_scheme}")
        print(f"N2 orientation scheme: {args.n2_orientation_scheme}")
        print(f"Max pair separation cutoff: {max_extent_A} A")
        print(f"Terminal sphere radius cutoff: {terminal_sphere_radius_A} A")
        print(f"Post-collision N2 detach distance: {n2_detach_distance_A} A")
        print(
            "Post-fragmentation target detach distance: "
            f"{target_fragment_detach_distance_A} A"
        )
        print(f"Shard: {args.shard_index}/{args.shard_count} ({len(assigned_cases)} assigned)")
        for case_type, target, energy, orientation in assigned_cases:
            steps = (
                (200 if args.quick else args.wall_steps)
                if case_type == "wall"
                else (100 if args.quick else args.n2_steps)
            )
            print(
                f"{case_type:4s} target={target:6s} energy={energy:g} eV "
                f"orientation={orientation:11s} steps={steps} dt={args.dt_fs} fs"
            )
        print(f"Assigned cases: {len(assigned_cases)} of {len(planned_cases)} total")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    calc = get_calculator(
        args.model,
        args.device,
        args.dtype,
        backend=args.calculator_backend,
        modal=args.model_modal,
        enable_flash=args.sevennet_enable_flash,
    )

    if args.calculator_backend == "mace-polar":
        package_version = __import__("mace").__version__
        calculator_name = "mace.calculators.mace_polar"
        charge_treatment = "Explicit total charge supplied through atoms.info."
    else:
        package_version = getattr(__import__("sevenn"), "__version__", "unknown")
        calculator_name = "sevenn.calculator.SevenNetCalculator"
        charge_treatment = (
            "SevenNet ASE calculator has no explicit net-charge input; EMI+, BF4-, and "
            "cluster labels are simulated with neutral-model forces."
        )

    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "model_modal": args.model_modal,
        "sevennet_enable_flash": args.sevennet_enable_flash,
        "calculator_backend": args.calculator_backend,
        "calculator": calculator_name,
        "calculator_package_version": package_version,
        "charge_treatment": charge_treatment,
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "rng_seed": RNG_SEED,
        "energies_eV": energies,
        "case_groups": args.case_groups,
        "n2_moving_species": args.n2_moving_species,
        "wall_orientation_scheme": args.wall_orientation_scheme,
        "n2_orientation_scheme": args.n2_orientation_scheme,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "energy_definition": "Total translational kinetic energy assigned to the moving subsystem.",
        "max_extent_A": max_extent_A,
        "terminal_sphere_radius_A": terminal_sphere_radius_A,
        "n2_detach_distance_A": n2_detach_distance_A,
        "target_fragment_detach_distance_A": target_fragment_detach_distance_A,
        "dt_fs": args.dt_fs,
        "n2_steps": args.n2_steps,
        "wall_steps": args.wall_steps,
        "notes": "MACE-POLAR-1 rerun: repulsive-wall EMI-BF4 and N2 impacts on EMI/BF4.",
    }
    metadata_name = "metadata.json" if args.shard_count == 1 else f"metadata_shard_{args.shard_index}.json"
    with open(RESULTS_DIR / metadata_name, "w") as f:
        json.dump(metadata, f, indent=2)

    all_results = []
    completed_cases = 0
    show_progress(completed_cases, len(assigned_cases), f"shard {args.shard_index}")
    wall_steps = 200 if args.quick else args.wall_steps
    n2_steps = 100 if args.quick else args.n2_steps

    if "wall" in args.case_groups:
        for energy in energies:
            for orientation in wall_orientations:
                case_key = ("wall", "EMIBF4", energy, orientation)
                if case_key not in assigned_case_set:
                    continue
                suffix = "" if orientation == "reference" else f"_O{orientation}"
                case_id = f"wall_EMIBF4_E{energy:04g}eV{suffix}"
                summary_path = RESULTS_DIR / case_id / f"{case_id}_summary.json"
                if args.resume and summary_path.exists():
                    print(f"Skipping completed {case_id}")
                    with open(summary_path) as f:
                        all_results.append(json.load(f))
                    completed_cases += 1
                    show_progress(completed_cases, len(assigned_cases), f"shard {args.shard_index}")
                    continue
                atoms, extra = setup_wall_case(energy, orientation=orientation)
                case = {
                    "case_id": case_id,
                    "case_type": "repulsive_wall",
                    "target": "EMIBF4",
                    "energy_eV": energy,
                    "calculator_wrapper": lambda base: RepulsiveWallCalculator(
                        base,
                        wall_position=0.0,
                        wall_stiffness=100.0,
                        max_extent_A=max_extent_A,
                        terminal_sphere_radius_A=terminal_sphere_radius_A,
                    ),
                    **extra,
                }
                print(f"Running {case['case_id']}")
                all_results.append(
                    run_md_case(
                        case,
                        atoms,
                        calc,
                        wall_steps,
                        dt_fs=args.dt_fs,
                        max_extent_A=max_extent_A,
                        terminal_sphere_radius_A=terminal_sphere_radius_A,
                    )
                )
                calc.reset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed_cases += 1
                show_progress(completed_cases, len(assigned_cases), f"shard {args.shard_index}")

    target_groups = []
    if "n2-emi" in args.case_groups:
        target_groups.append("EMI")
    if "n2-bf4" in args.case_groups:
        target_groups.append("BF4")
    if "n2-cluster-cation" in args.case_groups:
        target_groups.append("EMIBF4EMI")
    if "n2-cluster-anion" in args.case_groups:
        target_groups.append("EMIBF4BF4")
    for target in target_groups:
        for energy in energies:
            for orientation in n2_target_orientation_names(target, args.n2_orientation_scheme):
                case_key = ("n2", target, energy, orientation)
                if case_key not in assigned_case_set:
                    continue
                suffix = "" if orientation == "reference" else f"_O{orientation}"
                case_id = f"n2_{target}_E{energy:04g}eV{suffix}"
                summary_path = RESULTS_DIR / case_id / f"{case_id}_summary.json"
                if args.resume and summary_path.exists():
                    print(f"Skipping completed {case_id}")
                    with open(summary_path) as f:
                        all_results.append(json.load(f))
                    completed_cases += 1
                    show_progress(completed_cases, len(assigned_cases), f"shard {args.shard_index}")
                    continue
                n2_orientation = "perpendicular" if args.n2_orientation_scheme == "pilot3" else "random"
                atoms, extra = setup_n2_case(
                    target,
                    energy,
                    rng,
                    moving_species=args.n2_moving_species,
                    target_orientation=orientation,
                    n2_orientation=n2_orientation,
                )
                case = {
                    "case_id": case_id,
                    "case_type": "n2_impact",
                    "target": target,
                    "projectile": target if args.n2_moving_species == "target" else "N2",
                    "collision_partner": "N2",
                    "energy_eV": energy,
                    "impact_parameter_A": 0.0,
                    **extra,
                }
                if n2_detach_distance_A is None:
                    case["calculator_wrapper"] = lambda base: CenteredCalculator(
                        base,
                        max_extent_A=max_extent_A,
                        terminal_sphere_radius_A=terminal_sphere_radius_A,
                    )
                else:
                    case["calculator_wrapper"] = lambda base, target_indices=extra[
                        "target_indices"
                    ], n2_indices=extra["n2_indices"]: DetachableN2Calculator(
                        base,
                        target_indices=target_indices,
                        n2_indices=n2_indices,
                        detach_distance_A=n2_detach_distance_A,
                        target_fragment_detach_distance_A=target_fragment_detach_distance_A,
                        max_extent_A=max_extent_A,
                        terminal_sphere_radius_A=terminal_sphere_radius_A,
                    )
                print(f"Running {case['case_id']}")
                all_results.append(
                    run_md_case(
                        case,
                        atoms,
                        calc,
                        n2_steps,
                        dt_fs=args.dt_fs,
                        max_extent_A=max_extent_A,
                        terminal_sphere_radius_A=terminal_sphere_radius_A,
                    )
                )
                calc.reset()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                completed_cases += 1
                show_progress(completed_cases, len(assigned_cases), f"shard {args.shard_index}")

    summary_name = (
        "combined_summary.json"
        if args.shard_count == 1
        else f"combined_summary_shard_{args.shard_index}.json"
    )
    with open(RESULTS_DIR / summary_name, "w") as f:
        json.dump(all_results, f, indent=2)

    total = sum(r["elapsed_time_sec"] for r in all_results)
    print(f"Completed {len(all_results)} cases in {total:.1f} s ({total / 60:.2f} min)")


if __name__ == "__main__":
    main()
