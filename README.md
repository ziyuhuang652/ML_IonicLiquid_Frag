# EMI-BF4 wall-collision benchmark

This repository contains a publication-ready benchmark of neutral
EMI--BF4 wall-collision fragmentation for electrospray propulsion. The paper
compares published DFT/MM and ReaxFF results with local MACE-medium and
MACE-POLAR-1 workflows, focusing on product spectra, fragmentation onset,
product families, and computational cost.

## Folder Structure

```text
manuscript/   LaTeX source, compiled manuscript, and manuscript-used figures
data/         Retained datasets required by the active figures, tables, and validation
references/   BibTeX file, literature notes, and source-reference material
scripts/      Python analysis, plotting, validation, and simulation drivers
bash/         Shell entry points for simulation workflows
backup/       Archived drafts, unused figures, old scripts, unused data, and temp files
```

## Requirements

Use the conda environment as a starting point:

```bash
conda env create -f environment.yml
conda activate emibf4-benchmark
```

Core analysis requires Python, NumPy, pandas, matplotlib, ASE, and networkx.
Manuscript compilation requires TeX Live with `latexmk`, `elsarticle`,
`todonotes`, `siunitx`, and standard BibTeX support. Re-running simulations is
optional and requires the relevant external engines: LAMMPS with ReaxFF support
for ReaxFF and a current MACE/PyTorch installation for MACE-POLAR-1.

## Reproduce Figures And Tables

Run lightweight validation:

```bash
make validate
```

Regenerate the manuscript figures:

```bash
make figures
```

Compile the manuscript:

```bash
make manuscript
```

The main manuscript entry file is:

```text
manuscript/manuscript.tex
```

The active plotting script is:

```text
scripts/analysis/build_manuscript_figures.py
```

It writes the manuscript figures directly to `manuscript/figures/`.

## Optional Simulation Workflows

ReaxFF matched wall-collision cases:

```bash
export LAMMPS_EXECUTABLE=/path/to/lmp
JOBS=4 bash bash/run_reaxff_42.sh
```

MACE-POLAR-1 matched wall-collision cases:

```bash
DEVICE=cuda bash bash/run_mace_polar_42.sh
```

Simulation outputs are written under `results/`, which is ignored by Git.

## Data Policy

`data/` intentionally contains only the datasets needed by the current paper
figures, tables, and validation checks. Unused exploratory files, older figures,
raw trajectory sets not needed for active plots, and intermediate outputs were
moved to `backup/`. See `data/README.md` for retained dataset provenance and
`organization_report.md` for the move log and validation status.
