.PHONY: help install data train test lint format sam-validate sam-build sam-deploy clean

PYTHON ?= python
PIP ?= pip
SAM ?= sam

help:
	@echo "Serverless Anomaly Detection Pipeline - Management Commands"
	@echo ""
	@echo "  make install       Install production and development dependencies"
	@echo "  make data          Generate synthetic training data and sample CSV"
	@echo "  make train         Train Isolation Forest model on synthetic data"
	@echo "  make test          Run pytest suite with coverage"
	@echo "  make lint          Run flake8 and cfn-lint"
	@echo "  make format        Format code with black"
	@echo "  make sam-validate  Validate SAM CloudFormation template"
	@echo "  make sam-build     Build AWS SAM application artifacts"
	@echo "  make sam-deploy    Guided deployment to AWS with SAM"
	@echo "  make clean         Remove build, cache, and virtual environment files"

install:
	$(PIP) install -r requirements-dev.txt

data:
	$(PYTHON) scripts/generate_datasets.py

train:
	$(PYTHON) scripts/train_model.py

test:
	pytest -v --cov=src --cov-report=term-missing

lint:
	flake8 src tests scripts --max-line-length=120 --extend-ignore=E203,W503
	cfn-lint template.yaml

format:
	black --line-length=120 src tests scripts

sam-validate:
	$(SAM) validate --lint

sam-build:
	$(SAM) build --use-container

sam-deploy:
	$(SAM) deploy --guided

clean:
	rm -rf .pytest_cache .coverage htmlcov .aws-sam build dist *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
