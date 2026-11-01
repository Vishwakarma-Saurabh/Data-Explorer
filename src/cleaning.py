import pandas as pd


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if not constant_cols:
        return df.copy(), []
    return df.drop(columns=constant_cols).copy(), constant_cols


def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 0.50) -> tuple[pd.DataFrame, list[str]]:
    missing_rate = df.isnull().mean()
    to_drop = missing_rate[missing_rate > threshold].index.tolist()
    if not to_drop:
        return df.copy(), []
    return df.drop(columns=to_drop).copy(), to_drop


def fill_numeric_missing_with_median(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], int]:
    out = df.copy()
    num_cols = out.select_dtypes(include="number").columns
    medians = {}
    n_filled = 0
    for col in num_cols:
        n_missing = int(out[col].isnull().sum())
        if n_missing == 0:
            continue
        med = out[col].median()
        out[col] = out[col].fillna(med)
        medians[col] = med
        n_filled += n_missing
    return out, medians, n_filled


def clean_dataframe(df: pd.DataFrame, drop_constants: bool = True, missing_strategy: str = "none",) -> dict:
    result = df.copy()
    shape_before = result.shape
    dropped_constant = []
    dropped_high_missing = []
    medians_used = {}
    cells_filled = 0

    if drop_constants:
        result, dropped_constant = drop_constant_columns(result)

    if missing_strategy == "drop_rows":
        result = result.dropna().reset_index(drop=True)
    elif missing_strategy == "drop_cols_fill":
        result, dropped_high_missing = drop_high_missing_columns(result, threshold=0.50)
        result, medians_used, cells_filled = fill_numeric_missing_with_median(result)

    return {
        "df": result,
        "dropped_constant": dropped_constant,
        "dropped_high_missing": dropped_high_missing,
        "medians_used": medians_used,
        "cells_filled": cells_filled,
        "shape_before": shape_before,
        "shape_after": result.shape,
    }