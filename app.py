import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
from src.analysis import run_clustering, select_numeric_columns
import streamlit as st
import pandas as pd
from src.ingestion import read_csv_with_fallback, read_excel_with_fallback
from src.profiling import quick_insights, summarize_dtypes
from src.cleaning import clean_dataframe
from src.reporting import (
    build_profile_table,
    top_features_per_cluster,
    describe_cluster,
    build_markdown_report,
)

st.set_page_config(page_title="Auto EDA Studio", layout="wide")
st.title("Auto EDA Studio — Version - 5")


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

# --- Phase 4: Unsupervised analysis ---
st.subheader("Unsupervised Analysis (PCA + K-Means)")

numeric_df, dropped_non_numeric = select_numeric_columns(cleaned)

if dropped_non_numeric:
    st.info(
        f"Excluded {len(dropped_non_numeric)} non-numeric column(s) from analysis: "
        f"{dropped_non_numeric}. Categorical encoding arrives in Phase 4.5."
    )

if numeric_df.shape[1] < 2 or numeric_df.shape[0] < 4:
    st.warning(
        f"Need at least 2 numeric columns and 4 rows. "
        f"Got {numeric_df.shape[1]} numeric column(s), {numeric_df.shape[0]} row(s)."
    )
    st.stop()

with st.spinner("Running PCA + K-Means..."):
    try:
        analysis = run_clustering(numeric_df)
    except ValueError as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

# Silhouette score table
score_df = pd.DataFrame(analysis["scores"])
st.write("**Silhouette score per K** (higher is better; best in bold)")
best_row = score_df.loc[score_df["silhouette"].idxmax()]
score_df_display = score_df.copy()
score_df_display["silhouette"] = score_df_display["silhouette"].map(lambda v: f"{v:.3f}")
st.dataframe(score_df_display, use_container_width=True, hide_index=True)
st.caption(f"Auto-selected K = **{analysis['best_k']}** (highest silhouette = {best_row['silhouette']:.3f})")

# PCA scatter plot
ev = analysis["explained_variance_ratio"]
x_label = f"PC1 ({ev[0]:.1%} variance)"
y_label = f"PC2 ({ev[1]:.1%} variance)" if len(ev) > 1 else "PC2"

plot_df = pd.DataFrame(
    analysis["X_pca"], columns=["PC1", "PC2"] if analysis["X_pca"].shape[1] == 2 else ["PC1", "PC2"][:analysis["X_pca"].shape[1]]
)
plot_df["Cluster"] = analysis["labels"].astype(str)

fig = px.scatter(
    plot_df, x="PC1", y="PC2", color="Cluster",
    title=f"PCA projection — K={analysis['best_k']} clusters",
    opacity=0.75,
)
fig.update_layout(legend_title_text="Cluster")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Why 2D PCA (and not t-SNE / UMAP)?"):
    st.write(
        "PCA is linear and deterministic — running twice yields the same plot. "
        "t-SNE and UMAP are non-deterministic across versions and produce "
        "plots where **distance between clusters is not meaningful**. "
        "If PC1 + PC2 explain only a small fraction of variance, the 2D view "
        "is a lossy summary; treat cluster separation on this plot as a hint, "
        "not a proof."
    )
    st.write(f"Current PC1 + PC2 explain **{ev[0] + (ev[1] if len(ev) > 1 else 0):.1%}** of total variance.")


# --- Phase 5: Cluster profiling ---
st.subheader("Cluster Profiles")

numeric_cols = numeric_df.columns.tolist()
profile = build_profile_table(numeric_df, analysis["labels"], numeric_cols)
top_feats = top_features_per_cluster(profile, top_n=3)

# Full profile table (wide, but critical)
st.write("**Per-cluster means vs. overall**")
display_table = profile["cluster_means"].copy()
display_table["__overall__"] = profile["overall_mean"]
display_table["__size__"] = profile["cluster_sizes"]
st.dataframe(
    display_table.style.format("{:.3g}"),
    use_container_width=True,
)

# Per-cluster description cards
descriptions = {}
for cid in sorted(top_feats.keys()):
    size = int(profile["cluster_sizes"].loc[cid])
    desc = describe_cluster(cid, top_feats[cid], size=size)
    descriptions[cid] = desc
    st.markdown(f"**{desc}**")
    feats_table = pd.DataFrame(top_feats[cid])[["feature", "cluster_mean", "overall_mean", "z"]]
    feats_table["z"] = feats_table["z"].map(lambda v: f"{v:+.2f}")
    feats_table["cluster_mean"] = feats_table["cluster_mean"].map(lambda v: f"{v:.3g}")
    feats_table["overall_mean"] = feats_table["overall_mean"].map(lambda v: f"{v:.3g}")
    st.dataframe(feats_table, use_container_width=True, hide_index=True)

# --- Downloadable Markdown report ---
st.subheader("Download Report")
md = build_markdown_report(
    filename=uploaded.name,
    profile=profile,
    top_features=top_feats,
    descriptions=descriptions,
    silhouette_scores=analysis["scores"],
    best_k=analysis["best_k"],
    explained_variance=analysis["explained_variance_ratio"],
)
st.download_button(
    label="Download Markdown report",
    data=md,
    file_name=f"auto_eda_report_{uploaded.name.rsplit('.', 1)[0]}.md",
    mime="text/markdown",
)

with st.expander("Preview report"):
    st.markdown(md)


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
        except ImportError as er:
            st.error(f"Import failed: {er!r}")
            raise
        except Exception as e:
            st.error(f"Profiling failed: {type(e).__name__}: {e}")