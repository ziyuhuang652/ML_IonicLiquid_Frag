.PHONY: validate figures notebook manuscript all

validate:
	python scripts/analysis/validate_release.py

figures:
	python scripts/analysis/build_manuscript_figures.py

notebook:
	jupyter nbconvert --to notebook --execute scripts/analysis/comparison_analysis.ipynb \
		--output comparison_analysis.executed.ipynb \
		--ExecutePreprocessor.timeout=600

manuscript:
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex

all: validate figures notebook manuscript
