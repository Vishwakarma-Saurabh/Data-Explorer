import pandas as pd
import pytest
from src.reporting import (
    build_profile_table,
    top_features_per_cluster,
    describe_cluster,
    build_markdown_report,
)


@pytest.fixture
def toy_data():
    df = pd.DataFrame({
        "x": [10, 11, 12, 1, 2, 3],
        "y": [1, 2, 3, 10, 11, 12],
    })
    labels = [0, 0, 0, 1, 1, 1]
    return df, labels


def test_profile_table_shape_and_overall(toy_data):
    df, labels = toy_data
    p = build_profile_table(df, labels, ["x", "y"])
    assert list(p["cluster_means"].index) == [0, 1]
    assert p["overall_mean"]["x"] == pytest.approx(6.5)
    assert p["cluster_means"].loc[0, "x"] == pytest.approx(11.0)


def test_z_scores_sign(toy_data):
    df, labels = toy_data
    p = build_profile_table(df, labels, ["x", "y"])
    assert p["z_scores"].loc[0, "x"] > 0
    assert p["z_scores"].loc[0, "y"] < 0


def test_top_features_directions(toy_data):
    df, labels = toy_data
    p = build_profile_table(df, labels, ["x", "y"])
    tf = top_features_per_cluster(p, top_n=2)
    c0 = tf[0]
    assert c0[0]["direction"] in {"higher", "lower"}
    names = {f["feature"] for f in c0}
    assert names == {"x", "y"}


def test_describe_cluster_contains_count_and_direction(toy_data):
    df, labels = toy_data
    p = build_profile_table(df, labels, ["x", "y"])
    tf = top_features_per_cluster(p, top_n=2)
    desc = describe_cluster(0, tf[0], size=3)
    assert "Cluster 0" in desc
    assert "3 rows" in desc
    assert ("higher" in desc) or ("lower" in desc)


def test_markdown_report_has_required_sections(toy_data):
    df, labels = toy_data
    p = build_profile_table(df, labels, ["x", "y"])
    tf = top_features_per_cluster(p, top_n=2)
    descs = {cid: describe_cluster(cid, feats, size=3) for cid, feats in tf.items()}
    md = build_markdown_report(
        filename="toy.csv",
        profile=p,
        top_features=tf,
        descriptions=descs,
        silhouette_scores=[{"k": 2, "silhouette": 0.75, "inertia": 1.0}],
        best_k=2,
        explained_variance=[0.9, 0.1],
    )
    assert "# Auto EDA Report — toy.csv" in md
    assert "Auto-selected K: **2**" in md
    assert "## Cluster descriptions" in md
    assert "Cluster 0" in md