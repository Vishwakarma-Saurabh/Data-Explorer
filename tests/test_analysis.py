import numpy as np
import pandas as pd
import pytest
from src.analysis import (
    select_numeric_columns,
    fit_scaler_pca,
    score_k_range,
    run_clustering,
)


def _make_two_blob_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 2))
    b = rng.normal(loc=5.0, scale=0.5, size=(n // 2, 2))
    return pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])


def test_select_numeric_drops_strings():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "name": ["a", "b"]})
    num, dropped = select_numeric_columns(df)
    assert list(num.columns) == ["x", "y"]
    assert dropped == ["name"]


def test_fit_pca_rejects_one_column():
    with pytest.raises(ValueError):
        fit_scaler_pca(pd.DataFrame({"x": [1, 2, 3]}))


def test_fit_pca_explained_variance_sums_to_one():
    df = _make_two_blob_df()
    out = fit_scaler_pca(df)
    assert abs(sum(out["explained_variance_ratio"]) - 1.0) < 1e-6


def test_score_k_range_finds_k2_on_two_blobs():
    df = _make_two_blob_df(n=200)
    scaled = fit_scaler_pca(df)["X_scaled"]
    scores = score_k_range(scaled)
    best = max(scores, key=lambda r: r["silhouette"])
    assert best["k"] == 2, f"Expected k=2, got {best}"


def test_run_clustering_finds_two_clusters():
    df = _make_two_blob_df(n=200)
    out = run_clustering(df)
    assert out["best_k"] == 2
    assert len(set(out["labels"])) == 2
    assert out["X_pca"].shape == (200, 2)