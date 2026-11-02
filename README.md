# Auto EDA Studio

Upload a CSV or Excel dataset and get automated data-quality insights,
PCA-based visualization, and K-Means clustering with plain-language
cluster descriptions. No supervised ML — this is an **Auto-EDA + unsupervised
structure discovery** tool.

**Live demo:** https://auto-eda-studio.streamlit.app *(replace with your URL)*

> The live demo may take ~30 seconds to wake if idle. The Docker image is the
> reproducible deployment artifact; the Streamlit Cloud app is the quick-look demo.

## What it does

1. **Ingest** — CSV (UTF-8 / Latin-1, auto-delimiter) and Excel (.xlsx/.xlsm).
2. **Profile** — flags columns with >30% missing, constant columns, and
   high-cardinality ID-like strings.
3. **Clean** — user picks: drop constant columns, drop rows with missing,
   or drop >50%-missing columns and median-fill numeric.
4. **Cluster** — StandardScaler → PCA (2D) → K-Means with silhouette-based
   K selection (K=2..8). PCA scatter plot with explained variance.
5. **Explain** — per-cluster profile table (z-scores vs. overall mean),
   top distinguishing features, and a downloadable Markdown report.

## Architecture
repo/
├── app.py 
├── requirements.txt
├── packages.txt 
├── Dockerfile
├── Makefile
├── src/
│ ├── ingestion.py # File reading, encoding fallbacks
│ ├── profiling.py # Quick data-quality insights
│ ├── cleaning.py # User-controlled preprocessing
│ ├── analysis.py # StandardScaler, PCA, K-Means, silhouette
│ └── reporting.py # Cluster profiles + Markdown export
└── tests/

## Run locally
pip install -r requirements.txt

make run            # Start the Streamlit app
make test           # Run tests
make docker-build   # Build Docker image
make docker-run     # Run Docker container
make clean          # Remove cache files
