# Banking Churn & Recommendation — ML Models

Machine learning component of a school Big Data project: **customer retention (churn) and personalized product recommendation** for a bank/insurance use case.

This repository contains my individual contribution to a 3-person team project. My role covers the full ML layer: churn prediction, a real recommendation system, and a fallback segmentation system. The rest of the team builds the PySpark data pipeline + Streamlit dashboard, and the FastAPI + PostgreSQL backend that consumes these models.

## Context

**Target persona**: a bank customer advisor who needs, each morning, a prioritized list of at-risk customers plus a personalized product suggestion for each one.

**Deliverable (team-wide)**: a working web platform demo, not just a report.

## What's in here

| Module | Status | Description |
|---|---|---|
| `ml/churn/` | ✅ Done | Churn prediction — Random Forest classifier, calibrated risk thresholds, inference module |
| `ml/recommender/` | 🚧 In progress | Real recommendation system — collaborative filtering on the Santander dataset |
| `ml/fallback/` | ⏳ Not started | Segmentation-based fallback recommender (K-means + persona rules) |

## Tech stack

- Python, managed with [`uv`](https://github.com/astral-sh/uv)
- `pandas`, `scikit-learn` for modeling
- `joblib` for model persistence
- Jupyter notebooks for exploration, standalone `.py` modules for reusable inference code

## API contract

Model outputs are designed to match this Pydantic contract, consumed by the team's FastAPI backend:

```python
class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class ChurnResponse(BaseModel):
    client_id: int
    score: float          # 0.0 to 1.0
    risk_level: RiskLevel
    top_reasons: list[str]

class RecommendedProduct(BaseModel):
    name: str
    score: float

class RecommendResponse(BaseModel):
    client_id: int
    products: list[RecommendedProduct]
    source: str  # "collaborative_filtering" or "segmentation_fallback"
```

## Datasets

- **Churn Modelling** (Kaggle) — ~10,000 customers, used to train the churn classifier
- **Santander Product Recommendation** (Kaggle) — ~956,000 customers, 17 months of product history, sampled (~75k customers) for the recommender

These two datasets cover different (fictional/anonymized) customer populations — there is no public dataset combining real churn labels with real product history. This is a known, documented limitation of the project, not an oversight.

## Setup

```bash
uv sync
```

## Running the churn model

```bash
cd ml/churn
uv run jupyter lab   # open churn_eda.ipynb for the full exploration + training walkthrough
```

Inference on a single client:

```bash
cd ml/churn
uv run predict.py
```

## Churn model — key results

Two models compared on a stratified 80/20 split (10,000 customers, ~20% churn rate):

| Model | Recall (churn) | Precision (churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.70 | 0.39 | 0.777 |
| Random Forest | 0.61 | 0.63 | 0.857 |

Random Forest was selected for its stronger ROC-AUC and precision/recall balance. Since a missed churner (false negative) is costlier than a false alert for this use case, risk thresholds were calibrated below the default 0.5 cutoff:

- `high`: score ≥ 0.5
- `medium`: 0.25 ≤ score < 0.5
- `low`: score < 0.25

## Known limitations

- `top_reasons` is derived from global feature importance, not a per-client causal explanation (a proper solution would use SHAP values — not yet implemented)
- Churn and recommendation systems operate on disjoint customer populations (see Datasets section)
