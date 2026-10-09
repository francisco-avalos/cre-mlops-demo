.PHONY: setup data train monitor test api up down clean

setup:          ## install dependencies
	pip install -r requirements-dev.txt

data:           ## generate the synthetic CRE portfolio
	python data/generate_data.py

train:          ## train candidates, log to MLflow, register the winner
	python src/train.py

monitor:        ## run drift / skew / concept-drift / latency checks
	python src/monitor.py

test:           ## run the automated test suite
	pytest tests/ -v

ui:             ## open the local MLflow UI (http://localhost:5000)
	mlflow ui

api:            ## run the API locally (no Docker)
	uvicorn src.api:app --reload --port 8000

up:             ## build and start MLflow server + API in Docker
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf mlruns mlartifacts mlflow-store *.json drift_report.csv
