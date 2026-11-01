
import io
import pandas as pd


def read_csv_with_fallback(raw_bytes: bytes) -> tuple[pd.DataFrame, str]:
    attempts = [
        (dict(sep=",", encoding="utf-8-sig"), "UTF-8, comma"),
        (dict(sep=None, engine="python", encoding="utf-8-sig"), "UTF-8, auto-sep"),
        (dict(sep=None, engine="python", encoding="latin-1"), "Latin-1, auto-sep"),
    ]
    errors = []
    for kwargs, label in attempts:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), **kwargs)
            if df.shape[1] == 1 and "," in kwargs.get("sep", ","):
                errors.append(f"{label}: single column, wrong separator")
                continue
            return df, label
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}")
    raise ValueError("Could not read CSV. Tried:\n" + "\n".join(errors))


def read_excel_with_fallback(raw_bytes: bytes, filename: str) -> tuple[pd.DataFrame, str, list[str]]:
    if filename.lower().endswith(".xls"):
        raise ValueError("Legacy .xls not supported. Save as .xlsx and re-upload.")
    xls = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
    sheets = xls.sheet_names
    df = xls.parse(sheets[0])
    return df, f"Excel sheet '{sheets[0]}'", sheets