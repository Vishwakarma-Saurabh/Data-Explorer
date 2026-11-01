
import pandas as pd


def quick_insights(df: pd.DataFrame) -> list[dict]:
    insights = []

    # 1. High missing rate
    missing = df.isnull().mean()
    for col, rate in missing[missing > 0.30].items():
        insights.append({
            "severity": "warning",
            "column": col,
            "message": f"Missing rate {rate:.1%} — consider dropping or imputing.",
        })

    # 2. Constant columns
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            insights.append({
                "severity": "warning",
                "column": col,
                "message": "Constant column (single value) — provides no signal.",
            })

    # 3. High-cardinality strings (likely IDs)
    for col in df.select_dtypes(include="object").columns:
        ratio = df[col].nunique() / max(len(df), 1)
        if ratio > 0.80:
            insights.append({
                "severity": "info",
                "column": col,
                "message": f"High cardinality ({ratio:.0%} unique) — likely an ID, not a feature.",
            })

    return insights


def summarize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "non_null": df.notnull().sum().values,
        "unique": df.nunique(dropna=True).values,
    })