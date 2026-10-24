import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Auto EDA Studio", layout="wide")


def _read_csv(raw_bytes):
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


def _read_excel(raw_bytes, filename):
    if filename.lower().endswith(".xls"):
        raise ValueError("Legacy .xls not supported. Save as .xlsx and re-upload.")
    xls = pd.ExcelFile(io.BytesIO(raw_bytes), engine="openpyxl")
    sheets = xls.sheet_names
    df = xls.parse(sheets[0])
    return df, f"Excel sheet '{sheets[0]}'", sheets


@st.cache_data(show_spinner=False)
def load_data(raw_bytes, filename):
    """Cache on bytes, not file object. Returns (df, strategy, sheets)."""
    if filename.lower().endswith(".csv"):
        df, strategy = _read_csv(raw_bytes)
        return df, strategy, None
    df, strategy, sheets = _read_excel(raw_bytes, filename)
    return df, strategy, sheets


st.title("Auto EDA Studio — Phase 1")
st.caption("Upload a CSV or Excel file to preview it.")

uploaded = st.file_uploader("Choose file", type=["csv", "xlsx", "xlsm", "xls"])

if uploaded is None:
    st.info("Waiting for a file.")
    st.stop()

raw = uploaded.getvalue()

try:
    with st.spinner("Reading..."):
        df, strategy, sheets = load_data(raw, uploaded.name)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unexpected error: {type(e).__name__}: {e}")
    st.stop()

st.success(f"Loaded **{uploaded.name}** — {strategy}")

if sheets and len(sheets) > 1:
    st.info(f"This workbook has {len(sheets)} sheets: {sheets}. Only first is shown.")

col1, col2 = st.columns(2)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]}")

st.subheader("Data types")
st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]}), use_container_width=True, hide_index=True)

st.subheader("Preview (first 50 rows)")
st.dataframe(df.head(50), use_container_width=True)