#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-4}"
LAMMPS="${LAMMPS_EXECUTABLE:-$(command -v lmp || true)}"

if [[ -z "$LAMMPS" ]]; then
  echo "Set LAMMPS_EXECUTABLE to a LAMMPS binary with REAXFF support." >&2
  exit 2
fi

python "$ROOT/scripts/simulation/run_reaxff_wall_cases.py" \
  --lammps "$LAMMPS" \
  --results-dir "$ROOT/results/reaxff_42" \
  --energies 10 20 30 40 50 75 100 \
  --orientations dipole_px dipole_nx dipole_py dipole_ny dipole_pz dipole_nz \
  --steps 4000 \
  --jobs "$JOBS" \
  --resume
