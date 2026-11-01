import streamlit as st
import pandas as pd
from src.ingestion import read_csv_with_fallback, read_excel_with_fallback
from src.profiling import quick_insights, summarize_dtypes
from src.cleaning import clean_dataframe

st.set_page_config(page_title="Auto EDA Studio", layout="wide")
st.title("Auto EDA Studio — Phase 3")


@st.cache_data(show_spinner=False)
def load_data(raw_bytes: bytes, filename: str):
    if filename.lower().endswith(".csv"):
        df, strategy = read_csv_with_fallback(raw_bytes)
        return df, strategy, None
    df, strategy, sheets = read_excel_with_fallback(raw_bytes, filename)
    return df, strategy, sheets


@st.cache_data(show_spinner=False)
def cached_insights(df: pd.DataFrame) -> list[dict]:
    return quick_insights(df)


uploaded = st.file_uploader("Choose file", type=["csv", "xlsx", "xlsm", "xls"])
if uploaded is None:
    st.info("Waiting for a file.")
    st.stop()

raw = uploaded.getvalue()
try:
    with st.spinner("Reading..."):
        df, strategy, sheets = load_data(raw, uploaded.name)
except Exception as e:
    st.error(f"Read failed: {e}")
    st.stop()

st.success(f"Loaded **{uploaded.name}** — {strategy}")

# --- Profiling (Phase 2) ---
st.subheader("Data Quality Insights")
insights = cached_insights(df)
if not insights:
    st.success("No obvious data-quality red flags found in the first pass.")
else:
    for item in insights:
        msg = f"**{item['column']}**: {item['message']}"
        (st.warning if item["severity"] == "warning" else st.info)(msg)

# --- Cleaning (Phase 3) ---
st.subheader("Cleaning Options")
col_a, col_b = st.columns(2)
with col_a:
    drop_constants = st.checkbox("Drop constant columns", value=True)
with col_b:
    missing_strategy = st.radio(
        "Missing-value strategy",
        options=["none", "drop_rows", "drop_cols_fill"],
        format_func=lambda x: {
            "none": "Do nothing",
            "drop_rows": "Drop rows with any missing",
            "drop_cols_fill": "Drop >50% missing columns, median-fill numeric",
        }[x], index=0,
    )

result = clean_dataframe(df, drop_constants=drop_constants, missing_strategy=missing_strategy,)

shape_before = result["shape_before"]
shape_after = result["shape_after"]
st.write(
    f"**Shape:** {shape_before[0]:,} × {shape_before[1]}  →  "
    f"**{shape_after[0]:,} × {shape_after[1]}**"
)

if result["dropped_constant"]:
    st.info(f"Dropped constant columns ({len(result['dropped_constant'])}): {result['dropped_constant']}")
if result["dropped_high_missing"]:
    st.info(f"Dropped high-missing columns ({len(result['dropped_high_missing'])}): {result['dropped_high_missing']}")
if result["medians_used"]:
    st.info(f"Filled {result['cells_filled']} cells with column medians: {result['medians_used']}")

cleaned = result["df"]

st.subheader("Cleaned Data Preview")
st.dataframe(cleaned.head(50), use_container_width=True)

# --- Phase 2 summary on cleaned data ---
st.subheader("Column Summary (cleaned)")
st.dataframe(summarize_dtypes(cleaned), use_container_width=True, hide_index=True)

with st.expander("Full profiling report on cleaned data (slower)"):
    if st.button("Run ydata-profiling on cleaned data"):
        try:
            from ydata_profiling import ProfileReport
            with st.spinner("Profiling..."):
                profile = ProfileReport(cleaned, explorative=True, minimal=True)
                html = profile.to_html()
            st.components.v1.html(html, width=1000, height=600, scrolling=True)
        except ImportError:
            st.error("ydata-profiling not installed.")
        except Exception as e:
            st.error(f"Profiling failed: {type(e).__name__}: {e}")