.PHONY: validate benchmark bf4-173k-md all

validate:
	python scripts/analysis/validate_release.py

benchmark:
	python scripts/plotting/build_performance_benchmark.py

bf4-173k-md:
	python scripts/analysis/run_bf4_173k_bond_distribution.py

all: validate benchmark
