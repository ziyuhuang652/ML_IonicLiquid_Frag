#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE="${DEVICE:-cuda}"

python "$ROOT/scripts/simulation/run_mace_polar_cases.py" \
  --model polar-1-m \
  --device "$DEVICE" \
  --dtype float32 \
  --resume \
  --energies 10 20 30 40 50 75 100 \
  --case-groups wall \
  --wall-orientation-scheme paper6 \
  --max-extent-A 40 \
  --results-dir "$ROOT/results/mace_polar_42"
