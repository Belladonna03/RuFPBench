.PHONY: install pipeline collection rewrite quality annotation al

install:
	pip install -r requirements.txt

pipeline:
	python run_pipeline.py --config config.yaml

collection:
	python run_agent.py --agent collection --config config.yaml

rewrite:
	python run_agent.py --agent rewrite --config config.yaml --input data/raw/merged_raw.parquet

quality:
	python run_agent.py --agent quality --config config.yaml --input data/interim/rewrite.parquet

annotation:
	python run_agent.py --agent annotation --config config.yaml --input data/interim/clean.parquet

al:
	python run_agent.py --agent al --config config.yaml --input data/labeled/final_dataset.parquet
