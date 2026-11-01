import pandas as pd
import pytest
from src.cleaning import (
    drop_constant_columns,
    drop_high_missing_columns,
    fill_numeric_missing_with_median,
    clean_dataframe,
)


def test_drop_constant():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    out, dropped = drop_constant_columns(df)
    assert dropped == ["a"]
    assert out.shape == (3, 1)
    assert "a" not in out.columns


def test_drop_high_missing():
    df = pd.DataFrame({"a": [1, None, None], "b": [1, 2, 3]})
    out, dropped = drop_high_missing_columns(df, threshold=0.5)
    assert dropped == ["a"]
    assert out.shape == (3, 1)


def test_median_fill():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", None, "z"]})
    out, medians, n = fill_numeric_missing_with_median(df)
    assert medians == {"a": 2.0}
    assert n == 1
    assert out["a"].isnull().sum() == 0
    # Non-numeric untouched
    assert out["b"].isnull().sum() == 1


def test_clean_dataframe_does_not_mutate_input():
    df = pd.DataFrame({"const": [1, 1], "num": [1.0, None]})
    before = df.copy()
    out = clean_dataframe(df, drop_constants=True, missing_strategy="drop_cols_fill")
    pd.testing.assert_frame_equal(df, before), "Input must not be mutated"


def test_clean_dataframe_drop_rows():
    df = pd.DataFrame({"a": [1, 2, None], "b": [10, 20, 30]})
    out = clean_dataframe(df, drop_constants=False, missing_strategy="drop_rows")
    assert out["shape_after"] == (2, 2)