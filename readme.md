# MLOps Final Project

MLOps project for restaurant data analysis using Google Cloud BigQuery.

## Project Structure

```
mlops_final/
│
├── data/
│   ├── raw/                    # Original downloaded data from BigQuery
│   └── processed/              # Cleaned and processed data
│
├── notebooks/
│   └── exp.ipynb               # Exploratory data analysis
│
├── src/
│   ├── descarga.py             # Download data from BigQuery
│   ├── preprocess.py           # Data cleaning and preprocessing
│   ├── train.py                # Model training pipeline
│   └── evaluate.py             # Model evaluation and metrics
│
├── models/                     # Saved trained models
│
├── config.yaml                 # Configuration parameters
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image for deployment
├── Jenkinsfile                 # CI/CD pipeline definition
└── README.md
```

## Setup

1. Create virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Configure GCP credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

## Usage

```bash
python src/descarga.py      # Download data
python src/preprocess.py    # Clean data
python src/train.py         # Train models
python src/evaluate.py      # Evaluate models
```

## Project Phases

### Phase 1: GitHub Repository
- Data loading from BigQuery
- Exploratory data analysis
- Feature engineering
- Model training (minimum 2 models)
- Model evaluation
- Docker deployment

### Phase 2: Jenkins Pipeline
- Automated CI/CD pipeline
- Testing integration
- Notifications

### Phase 3: Vertex AI Deployment
- AutoML Tabular training
- Custom Vertex Pipelines
- Performance metrics and comparison
