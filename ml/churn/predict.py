"""
Churn inference module.

Wraps the trained Random Forest model so it can be called on a single
new client and returns output matching the ChurnResponse contract:

    class RiskLevel(str, Enum):
        low = "low"
        medium = "medium"
        high = "high"

    class ChurnResponse(BaseModel):
        client_id: int
        score: float
        risk_level: RiskLevel
        top_reasons: list[str]
"""

from pathlib import Path

import joblib
import pandas as pd

# Resolve paths relative to this file's own location, not the working
# directory the script happens to be launched from. Model files live one
# level up, in ml/, since they're shared assets rather than churn-specific
# source code.
_MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = _MODULE_DIR.parent / "churn_model.joblib"
COLUMNS_PATH = _MODULE_DIR.parent / "churn_model_columns.joblib"

# Thresholds decided from threshold-sweep analysis (see churn_eda.ipynb).
# A false negative (missed churner) is costlier than a false positive here,
# so thresholds are set below the default 0.5 to favor catching more risk.
HIGH_RISK_THRESHOLD = 0.5
MEDIUM_RISK_THRESHOLD = 0.25

_model = None
_expected_columns = None


def _load_model():
    """Lazy-load the model and expected feature columns once per process."""
    global _model, _expected_columns
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _expected_columns = joblib.load(COLUMNS_PATH)
    return _model, _expected_columns


def _preprocess(raw_client: dict) -> pd.DataFrame:
    """
    Apply the exact same encoding used at training time.

    raw_client is expected to have the original (pre-encoding) fields, e.g.:
    {
        "CreditScore": 650, "Geography": "Germany", "Gender": "Female",
        "Age": 45, "Tenure": 3, "Balance": 120000.0, "NumOfProducts": 2,
        "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 80000.0
    }
    """
    _, expected_columns = _load_model()

    df = pd.DataFrame([raw_client])
    df_encoded = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Reindex to match training columns exactly: adds any missing one-hot
    # column as 0 (e.g. client is French -> Geography_Germany and
    # Geography_Spain are both absent from this row's dummies, but the
    # model still expects those columns to exist) and drops/reorders
    # anything unexpected. This is the step that prevents silent
    # train/inference mismatches described above.
    df_aligned = df_encoded.reindex(columns=expected_columns, fill_value=0)
    return df_aligned


def _risk_level(score: float) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "high"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _top_reasons(model, client_row: pd.DataFrame, top_n: int = 3) -> list[str]:
    """
    Simple, explainable heuristic: rank this client's features by the
    model's global feature_importances_, then report the ones where this
    specific client's value looks notable (not a rigorous per-client
    explanation like SHAP would give -- just a readable first pass).
    """
    importances = pd.Series(model.feature_importances_, index=client_row.columns)
    top_features = importances.sort_values(ascending=False).head(top_n).index

    reasons = []
    for feature in top_features:
        value = client_row.iloc[0][feature]
        reasons.append(f"{feature}={value}")
    return reasons


def predict_churn(client_id: int, raw_client: dict) -> dict:
    """
    Main entry point. Returns a dict matching the ChurnResponse schema.
    """
    model, _ = _load_model()
    client_row = _preprocess(raw_client)

    score = float(model.predict_proba(client_row)[0][1])

    return {
        "client_id": client_id,
        "score": round(score, 4),
        "risk_level": _risk_level(score),
        "top_reasons": _top_reasons(model, client_row),
    }


if __name__ == "__main__":
    # Quick manual check -- run with: uv run python predict.py
    example_client = {
        "CreditScore": 650,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 3,
        "Balance": 120000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 80000.0,
    }
    result = predict_churn(client_id=99999, raw_client=example_client)
    print(result)
