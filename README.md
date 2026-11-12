# Session-Based GNN Recommendation System

A course project for next-item recommendation from anonymous browsing sessions, combining graph neural network models with reproducible ML pipelines, model serving, and monitoring.

## My contribution

**Tran Anh Tuan - Team Lead / Solution Architect | Team of 5 | February - May 2026**

- Led solution architecture, task planning, and testing across the recommendation workflow.
- Coordinated data preparation, GNN training and evaluation, model comparison, and serving interfaces.
- Planned the integration of experiment tracking, data versioning, CI/CD, and operational monitoring.

This is a portfolio fork of the [team repository](https://github.com/tlong-ds/recsys-group-project).
Models, pipelines, and deployment assets are a shared team effort.

## What is in the repository

- **Session-based GNNs:** PyTorch implementations of SR-GNN variants, GGNN, and TAGNN for next-item ranking.
- **Reproducible experiments:** DVC data/training pipelines, version comparisons, and MLflow experiment tracking.
- **Evaluation and selection:** Hit Rate@K and MRR@K, model-matrix evaluation, selection, retraining, and promotion stages.
- **Serving:** FastAPI recommendation endpoints and a React/TypeScript interface.
- **Observability:** Prometheus/Grafana configuration and Evidently monitoring workflows.
- **Delivery assets:** Docker Compose, GitHub Actions, Kubernetes manifests, and Terraform configuration.

Docker Compose supports the local stack; Kubernetes manifests and Terraform configuration describe the cloud deployment workflow.

## Pipeline

```text
Session data -> validation / preprocessing -> versioned DVC datasets
  -> GNN model matrix -> ranking evaluation -> model selection / retraining
  -> model artifact -> FastAPI -> recommendation interface

MLflow: parameters, metrics, artifacts
Prometheus / Grafana: serving telemetry
Evidently: monitoring and drift reports
```

## Technology Stack

### Backend (Python 3.10+)
- **Core Framework:** [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/)
- **Data Processing:** [Polars](https://pola.rs/), [Pandas](https://pandas.pydata.org/), [PyArrow](https://arrow.apache.org/docs/python/), [Pandera](https://pandera.readthedocs.io/)
- **Utilities:** [Loguru](https://github.com/Delgan/loguru), [Pydantic](https://docs.pydantic.dev/), [PyYAML](https://pyyaml.org/)

### Frontend
- **Framework:** [React](https://react.dev/), [TypeScript](https://www.typescriptlang.org/), [Vite](https://vitejs.dev/)
- **State & Animation:** [Zustand](https://zustand-demo.pmnd.rs/), [Framer Motion](https://www.framer.com/motion/), [D3.js](https://d3js.org/)
- **Deployment:** [Cloudflare Pages](https://pages.cloudflare.com/)

### Machine Learning
- **Frameworks:** [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) (SR-GNN, GGNN, TAGNN), [Scikit-learn](https://scikit-learn.org/)
- **Experiment Tracking:** [MLflow](https://mlflow.org/), [DagsHub](https://dagshub.com/)

### DevOps & MLOps
- **Containerization:** [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)
- **Orchestration:** [Kubernetes](https://kubernetes.io/) (AWS EKS), [Kustomize](https://kustomize.io/)
- **Infrastructure as Code:** [Terraform](https://www.terraform.io/)
- **Data Versioning:** [DVC](https://dvc.org/) (Data Version Control) with S3 backend
- **CI/CD:** [GitHub Actions](https://github.com/features/actions), [GitHub Container Registry](https://github.com/features/packages)
- **Monitoring:** [Prometheus](https://prometheus.io/), [Grafana](https://grafana.com/), [Evidently](https://www.evidentlyai.com/) (Model Drift & Quality)

### Testing & Quality
- **Testing:** [Pytest](https://docs.pytest.org/), [Unittest](https://docs.python.org/3/library/unittest.html)
- **Linting & Formatting:** [Ruff](https://beta.ruff.rs/docs/), [ESLint](https://eslint.org/)
- **Static Analysis:** [Mypy](http://mypy-lang.org/), [Bandit](https://bandit.readthedocs.io/), [Pip-audit](https://github.com/pypa/pip-audit)

## Repository Structure

```text
recsys-group-project/
├── .github/workflows/      # CI: lint, test, package checks
├── configs/                # YAML configs for data, model, training, serving
├── data/                   # raw/, interim/, processed/ datasets
├── deployment/             # K8s, MLflow, monitoring assets, Terraform
├── diagrams/               # Architecture and MLOps diagrams
├── docs/                   # Problem statement and team contracts
├── frontend/               # React Vite SPA for user interface
├── metrics/                # Local metrics tracking and model selection JSONs
├── models/trained/         # Local model registry / exported artifacts
├── notebooks/              # EDA notebooks
├── pipelines/              # DVC pipelines (data, monitoring, training)
├── reports/                # Monitoring and evaluation reports
├── scripts/                # Utility scripts (e.g., DVC lock splitting)
├── src/recsys/             # Application package
├── tests/                  # Unit and smoke tests
├── Dockerfile              # Backend container definition
├── docker-compose.yaml     # Local stack composition
├── pyproject.toml          # Python package config
├── requirements.txt        # Python dependencies
└── README.md
```

## Quick Start

### Backend

```bash
# Install package with development dependencies
pip install -e .[dev]

# Run tests
python -m unittest discover -s tests -p "test_*.py"

# Process data, train model, and serve the API locally
recsys-process-data --stage all --config configs/data_config.yaml --params params.yaml
recsys-train --data-config configs/data_config.yaml --model-config configs/model_config.yaml --training-config configs/training_config.yaml --params params.yaml
recsys-serve --config configs/serving_config.yaml
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Configuration Ownership

Runtime config is merged in this order:
1. `configs/*.yaml` defaults
2. `params.yaml` overrides (**experiment knobs only**)

Use `params.yaml` for DVC experiment tuning only (model/data/training hyperparameters). Keep metadata and runtime settings (paths, registry, MLflow URI, DagsHub repo, report destinations) in `configs/*_config.yaml`.

## MLOps & DVC Pipelines

### Versioned Data Pipelines (V1/V2)

Each data version has its own params file and artifact directory so it can be tracked independently in DVC. The repo provides a closed-loop comparison target that runs data build + train/eval + drift monitoring + aggregation in one pipeline.

```bash
# Run the full version comparison pipeline
dvc repro pipelines/monitoring/dvc.yaml:compare_data_versions

# Build data versions independently (no training)
dvc repro pipelines/data/dvc.yaml:data_version_v1
dvc repro pipelines/data/dvc.yaml:data_version_v2
```

Version definitions:
- V1 strict filter: `configs/data_versions/v1_strict_filter.yaml`
- V2 sliding-window safety: `configs/data_versions/v2_sliding_window.yaml`

Generated artifacts are stored in `data/versions/`, `models/trained/`, and `metrics/`.

### Model Matrix Training

Model profiles are defined in `configs/model_profiles/*.yaml` (e.g., `srgnn`, `ggnn`, `tagnn`).

Run fast profiles (default iteration loop):
```bash
dvc repro pipelines/training/dvc.yaml:train_matrix pipelines/training/dvc.yaml:evaluate_matrix
```

Run long-running profiles:
```bash
dvc repro pipelines/training/dvc.yaml:train_matrix_slow pipelines/training/dvc.yaml:evaluate_matrix_slow
```

Final production flow retrains the selected winner on merged train+val data:
```bash
dvc repro pipelines/training/dvc.yaml:select_best_model pipelines/training/dvc.yaml:retrain_selected_model pipelines/training/dvc.yaml:promote_retrained_model
```

### Artifact Commit Hygiene

Training outputs are intentionally split into DVC-managed model artifacts and local-only experiment metrics.
Before committing, ensure only source/config/lock changes are present:

```bash
git status --short
# Expected: changes to pipelines/*/dvc.yaml, pipelines/*/dvc.lock, code, config, docs
```

After training and before `dvc push`, confirm new cache objects:
```bash
dvc status -c
dvc push
```

If migrating from the legacy monolithic `dvc.lock`, split the lock state first without recomputing artifacts:
```bash
python scripts/split_dvc_lock.py
```

## Experiment Tracking & Monitoring

### Optional DagsHub MLflow Tracking

To track experiments via DagsHub/MLflow, enable it in `configs/training_config.yaml`:
```yaml
mlflow:
  enabled: true
  dagshub:
    enabled: true
    token_env_var: DAGSHUB_USER_TOKEN
    repo_owner: lytlong.pers
    repo_name: recsys-group-project
```

Export your token before running:
```bash
export DAGSHUB_USER_TOKEN=<your_personal_access_token>
```

### MLflow System Metrics

Enable MLflow system metrics collection by config to track CPU/memory/GPU usage:
```yaml
mlflow:
  system_metrics:
    enabled: true
    sampling_interval: 10
    samples_before_logging: 1
```

Install host dependencies:
```bash
pip install psutil
# Optional GPU metrics: pip install nvidia-ml-py
```

## CI/CD Deployment

GitHub Actions publishes container images to the GitHub Container Registry (GHCR) after successful CI runs triggered by pushes to `main`.
- **Image:** `ghcr.io/<owner>/<repo>`
- **Tags:** `main` and `sha-<short_sha>`

For Kubernetes deployment assets, refer to the `deployment/` directory.