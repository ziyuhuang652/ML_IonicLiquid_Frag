# Data Inventory

This folder keeps only datasets required by the active manuscript figures,
tables, validation checks, or setup rendering. Older, exploratory, duplicated,
or unused datasets are archived under `backup/data/`.

## Retained Datasets

| File | Contents | Used by | Supports |
| --- | --- | --- | --- |
| `processed/mass_spectra_standardized.csv` | Standardized DFT/MM, ReaxFF, MACE-medium, and MACE-POLAR-1 mass-spectrum peak tables | `scripts/analysis/build_manuscript_figures.py` | Fig. 2, selected 10/40/100 eV spectra |
| `processed/wall_cases_normalized.csv` | Normalized 42-case MACE-medium and 42-case MACE-POLAR-1 wall-collision outcomes | `build_manuscript_figures.py`, `validate_release.py` | Fig. 3, validation counts |
| `processed/hf_trajectory_occurrence.csv` | Trajectory-level HF occurrence flags for MACE and ReaxFF summaries | `build_manuscript_figures.py`, `validate_release.py` | Validation and internal outcome-map joins |
| `processed/reaxff_case_summary.csv` | Matched 42-case ReaxFF wall-collision summary | `build_manuscript_figures.py`, `validate_release.py` | Fig. 3, Fig. 4, validation counts |
| `processed/wall_products.csv` | Product formula table from MACE terminal states | `build_manuscript_figures.py` | Fig. 4 product-family comparison |
| `literature/dft_energy_resolved_results.csv` | Energy-resolved DFT/MM regime and product conclusions extracted from the reference paper | `build_manuscript_figures.py` | Fig. 3 DFT regime strip, Fig. 4 DFT product presence |
| `benchmarks/resource_benchmark.csv` | Reported and measured per-trajectory wall-time and resource values | `validate_release.py`; manuscript table | Runtime benchmark table |
| `raw/mace_medium_42/run_E10_Y-90_Z-90/E10_Y-90_Z-90.traj` | One archived MACE-medium initial trajectory containing the explicit Au slab and EMI--BF4 geometry | `build_manuscript_figures.py` | Fig. 1 collision setup rendering |

## Archived Data

`backup/data/` contains unused digitized spectra, older processed comparison
tables, unused literature-derived CSVs, full archived MACE trajectories, full
MACE-POLAR raw outputs, and raw ReaxFF case directories. They were moved rather
than deleted so the analysis can be expanded later if the manuscript scope
changes.
