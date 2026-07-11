#!/usr/bin/env python3
"""Validate the compact release tables and headline manuscript values."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    cases = pd.read_csv(DATA / "processed/wall_cases_normalized.csv")
    hf = pd.read_csv(DATA / "processed/hf_trajectory_occurrence.csv")
    reax = pd.read_csv(DATA / "processed/reaxff_case_summary.csv")
    benchmark = pd.read_csv(DATA / "benchmarks/resource_benchmark.csv")

    require(len(cases) == 84, f"Expected 84 MACE cases, found {len(cases)}")
    require(
        cases.groupby("model").size().to_dict()
        == {"MACE-medium": 42, "MACE-polar": 42},
        "Unexpected MACE model counts",
    )
    require(len(reax) == 42, f"Expected 42 ReaxFF cases, found {len(reax)}")

    observed = (
        hf.groupby("model")
        .agg(any_HF=("any_HF", "sum"), terminal_HF=("terminal_HF", "sum"))
        .astype(int)
        .to_dict("index")
    )
    require(observed["MACE-medium"] == {"any_HF": 11, "terminal_HF": 4}, str(observed))
    require(observed["MACE-polar"] == {"any_HF": 19, "terminal_HF": 17}, str(observed))
    require(observed["ReaxFF"] == {"any_HF": 0, "terminal_HF": 0}, str(observed))

    dft_hours = float(
        benchmark.loc[benchmark.method == "DFT/QM-MM", "wall_time_hours_per_2ps_trajectory"].iloc[0]
    )
    require(dft_hours == 1440.0, f"Unexpected DFT upper bound: {dft_hours}")
    print("Release validation passed: 126 cases, HF totals, and runtime benchmark.")


if __name__ == "__main__":
    main()
