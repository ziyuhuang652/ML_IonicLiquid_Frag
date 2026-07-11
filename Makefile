.PHONY: validate benchmark all

validate:
	python scripts/analysis/validate_release.py

benchmark:
	python scripts/plotting/build_performance_benchmark.py

all: validate benchmark
