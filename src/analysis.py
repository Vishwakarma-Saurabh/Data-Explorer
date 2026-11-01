from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MAX_SILHOUETTE_ROWS = 10_000
K_RANGE = range(2, 9)


def select_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    num = df.select_dtypes(include="number")
    dropped = [c for c in df.columns if c not in num.columns]
    return num, dropped


def fit_scaler_pca(num_df: pd.DataFrame, n_components: int = 2) -> dict:
    if num_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for unsupervised analysis.")
    if num_df.shape[0] < 4:
        raise ValueError("Need at least 4 rows for stable clustering.")

    X = num_df.fillna(num_df.median()).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    return {
        "X_scaled": X_scaled,
        "X_pca": X_pca,
        "scaler": scaler,
        "pca": pca,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def score_k_range(X_scaled: np.ndarray) -> list[dict]:
    n = X_scaled.shape[0]
    if n > MAX_SILHOUETTE_ROWS:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, MAX_SILHOUETTE_ROWS, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    results = []
    for k in K_RANGE:
        if k >= X_sample.shape[0]:
            break
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sample)
        if len(set(labels)) < 2:
            continue
        score = float(silhouette_score(X_sample, labels))
        results.append({"k": k, "silhouette": score, "inertia": float(km.inertia_)})
    return results


def run_clustering(num_df: pd.DataFrame, k: int | None = None) -> dict:
    scaled = fit_scaler_pca(num_df)
    scores = score_k_range(scaled["X_scaled"])

    if not scores:
        raise ValueError("Could not evaluate any K value — dataset too small.")

    if k is None:
        best_k = max(scores, key=lambda r: r["silhouette"])["k"]
    else:
        best_k = int(k)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(scaled["X_scaled"])

    return {
        "X_pca": scaled["X_pca"],
        "labels": labels,
        "best_k": best_k,
        "scores": scores,
        "explained_variance_ratio": scaled["explained_variance_ratio"],
        "scaler": scaled["scaler"],
        "pca": scaled["pca"],
    }