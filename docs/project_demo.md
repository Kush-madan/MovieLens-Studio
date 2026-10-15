# Portfolio Demo Sequence

This sequence is designed for a five-minute final-year project or interview demonstration.

## 1. Dataset and pipeline overview

Show the repository tree and explain the Bronze/Silver/Gold contract. Point to `config/config.yaml`, then describe the six ordered pipeline tasks and why each stage writes Parquet.

Useful facts from the real dataset:

- 6,040 users
- 3,883 movies
- 1,000,209 ratings
- Date range: April 26, 2000 to February 28, 2003

## 2. Analytics dashboard

Launch:

```powershell
python -m streamlit run dashboard\app.py --server.port 8502
```

Open Overview, then Movie Analytics. Explain that the KPI cards, charts, and tables read Gold JSON/Parquet artifacts. Show sparsity/density and a genre or ratings-over-time chart.

## 3. Personalized recommendations

Open Personalized Recommendations and enter a known user ID such as `1`. Explain that ALS recommendations contain predicted scores and that already-rated movies are removed. Enter an unknown ID such as `999999` to show the explicit `popularity_fallback` source.

## 4. Model metrics

Open Model Performance and point to the saved RMSE `0.869412`, MAE `0.696190`, training/test row counts, and evaluated prediction count. Explain why these are rating-prediction metrics rather than proof of ranking quality.

## 5. Data quality and orchestration

Open Pipeline & Data Quality to show ingestion counts, Silver rejection counters, Gold row counts, and artifact availability. Then show `airflow/dags/movie_recsys_dag.py` and its final validation task. State clearly that Airflow has not been executed on the Windows development machine and is intended for WSL2 or Docker.

## Suggested screenshots

Capture the six dashboard views as:

```text
docs/screenshots/overview.png
docs/screenshots/movie-analytics.png
docs/screenshots/recommendations.png
docs/screenshots/movie-explorer.png
docs/screenshots/model-performance.png
docs/screenshots/pipeline-quality.png
```

These files are not included yet; capture them from a running local dashboard before publishing a portfolio repository.