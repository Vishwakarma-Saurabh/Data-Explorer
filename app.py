
import pandas as pd
import streamlit as st
from src.ingestion import read_csv_with_fallback, read_excel_with_fallback
from src.profiling import quick_insights, summarize_dtypes

st.set_page_config(page_title="Auto EDA Studio", layout="wide")
st.title("Auto EDA Studio — Phase 2")


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

# --- Quick insights ---
st.subheader("Data Quality Insights")
insights = cached_insights(df)
if not insights:
    st.success("No obvious data-quality red flags found in the first pass.")
else:
    for item in insights:
        msg = f"**{item['column']}**: {item['message']}"
        if item["severity"] == "warning":
            st.warning(msg)
        else:
            st.info(msg)

# --- Dtype summary table ---
st.subheader("Column Summary")
st.dataframe(summarize_dtypes(df), use_container_width=True, hide_index=True)

# --- Preview ---
st.subheader("Preview (first 50 rows)")
st.dataframe(df.head(50), use_container_width=True)

# --- Optional full report ---
with st.expander("Generate full profiling report (slower, deeper)"):
    if st.button("Run ydata-profiling"):
        try:
            from ydata_profiling import ProfileReport
            with st.spinner("Profiling all columns..."):
                profile = ProfileReport(df, explorative=True, minimal=True)
                html = profile.to_html()
            st.components.v1.html(html, width=1000, height=600, scrolling=True)
        except ImportError:
            st.error("ydata-profiling not installed. Run: pip install ydata-profiling")
        except Exception as e:
            st.error(f"Profiling failed: {type(e).__name__}: {e}")