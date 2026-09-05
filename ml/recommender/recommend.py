"""
Recommender inference module.

Wraps the item-based collaborative filtering similarity matrix so it can
be called on a single customer and returns output matching the
RecommendResponse contract:

    class RecommendedProduct(BaseModel):
        name: str
        score: float

    class RecommendResponse(BaseModel):
        client_id: int
        products: list[RecommendedProduct]
        source: str  # "collaborative_filtering" or "segmentation_fallback"
"""

from pathlib import Path

import joblib

# Resolve paths relative to this file's own location, not the working
# directory the script happens to be launched from (see predict.py in
# ml/churn/ for the same reasoning).
_MODULE_DIR = Path(__file__).resolve().parent
SIMILARITY_PATH = _MODULE_DIR.parent / "recommender_similarity.joblib"
PRODUCT_MATRIX_PATH = _MODULE_DIR.parent / "recommender_product_matrix.joblib"

_similarity_df = None
_product_matrix = None


def _load_artifacts():
    """Lazy-load the similarity matrix and product matrix once per process."""
    global _similarity_df, _product_matrix
    if _similarity_df is None:
        _similarity_df = joblib.load(SIMILARITY_PATH)
        _product_matrix = joblib.load(PRODUCT_MATRIX_PATH)
    return _similarity_df, _product_matrix


def recommend_products(client_id: int, top_n: int = 3) -> dict:
    """
    Main entry point. Returns a dict matching the RecommendResponse schema.

    Returns an empty products list (source still "collaborative_filtering")
    when:
      - the client_id is not in the sampled product matrix, or
      - the client holds zero products yet (no signal to recommend from).
    Both cases are exactly where the fallback segmentation system
    (ml/fallback/) is expected to take over -- this function does not
    call the fallback itself, that decision belongs to the caller (API).
    """
    similarity_df, product_matrix = _load_artifacts()

    if client_id not in product_matrix.index:
        return {
            "client_id": int(client_id),
            "products": [],
            "source": "collaborative_filtering",
        }

    customer_products = product_matrix.loc[client_id]
    owned = customer_products[customer_products == 1].index.tolist()

    if not owned:
        return {
            "client_id": int(client_id),
            "products": [],
            "source": "collaborative_filtering",
        }

    # Score each not-yet-owned product by summing its similarity to every
    # product the client already holds.
    scores = similarity_df[owned].sum(axis=1)
    scores = scores.drop(owned)
    top_scores = scores.sort_values(ascending=False).head(top_n)

    products = [
        {"name": name, "score": round(float(score), 4)}
        for name, score in top_scores.items()
    ]

    return {
        "client_id": int(client_id),
        "products": products,
        "source": "collaborative_filtering",
    }


if __name__ == "__main__":
    # Quick manual check -- run with: uv run recommend.py
    _, product_matrix = _load_artifacts()

    # Pick a client that actually holds at least one product, rather than
    # an arbitrary index -- ~25% of sampled clients hold zero products,
    # so index[0] alone isn't a reliable test case.
    clients_with_products = product_matrix[product_matrix.sum(axis=1) > 0].index
    example_client_id = clients_with_products[0]

    print(f"Testing with client_id={example_client_id}")
    print(
        f"Products held: {product_matrix.loc[example_client_id][product_matrix.loc[example_client_id] == 1].index.tolist()}"
    )

    result = recommend_products(client_id=example_client_id, top_n=3)
    print(result)
