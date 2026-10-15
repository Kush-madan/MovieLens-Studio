"""Streamlit dashboard for the Movie Recommendation System."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dashboard.components import apply_style, metric_row, missing_artifact, page_header  # noqa: E402
from dashboard.data_loader import (  # noqa: E402
    DashboardDataError,
    artifact_availability,
    available_genres,
    dashboard_config,
    filter_movies,
    filter_recommendations,
    load_analytics_summary,
    load_gold_table,
    load_ingestion_stats,
    load_model_metrics,
    load_quality_stats,
    load_recommendations,
)


st.set_page_config(
    page_title="MovieLens Recommendation Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_style()


@st.cache_data(ttl=300, show_spinner=False)
def cached_summary():
    return load_analytics_summary()


@st.cache_data(ttl=300, show_spinner=False)
def cached_metrics():
    return load_model_metrics()


@st.cache_data(ttl=300, show_spinner=False)
def cached_table(name: str):
    return load_gold_table(name)


@st.cache_data(ttl=300, show_spinner=False)
def cached_recommendations():
    return load_recommendations()


def format_number(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:,.3f}"
    return f"{int(value):,}" if isinstance(value, (int, float)) else str(value)


def format_average(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value):.2f}"


def format_percentage(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return f"{float(value) * 100:.2f}%"


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.strftime("%b %d, %Y")


def overview_page() -> None:
    page_header("01 / Overview", "MovieLens at a glance", "Actual Gold analytics from the processed MovieLens 1M dataset.")
    summary = cached_summary()
    if summary is None:
        missing_artifact("Gold analytics summary", "Run the analytics stage to generate `data/gold/analytics_summary.json`.")
        return
    metric_row([
        ("Users", format_number(summary.get("total_users")), None),
        ("Movies", format_number(summary.get("total_movies")), None),
        ("Ratings", format_number(summary.get("total_ratings")), None),
        ("Global average", format_average(summary.get("global_average_rating")), "Mean rating across all Silver ratings."),
    ])
    st.write("")
    metric_row([
        ("Sparsity", format_percentage(summary.get("sparsity")), "1 minus observed ratings divided by all user/movie pairs."),
        ("Density", format_percentage(summary.get("density")), "Observed ratings divided by all user/movie pairs."),
        ("Date start", format_date(summary.get("date_range", {}).get("start")), None),
        ("Date end", format_date(summary.get("date_range", {}).get("end")), None),
    ])
    st.subheader("Gold outputs")
    st.dataframe(pd.DataFrame([{"table": key, "rows": value} for key, value in summary.get("row_counts", {}).items()]), use_container_width=True, hide_index=True)


def movie_analytics_page() -> None:
    page_header("02 / Movie Analytics", "What audiences actually rated", "Explore popularity, rating quality, genres, and rating volume over time.")
    popular = cached_table("most_popular_movies")
    top_rated = cached_table("top_rated_movies")
    genre = cached_table("genre_popularity")
    distribution = cached_table("rating_distribution")
    over_time = cached_table("ratings_over_time")
    if popular is None or top_rated is None:
        missing_artifact("Movie analytics tables", "Run the analytics stage to generate Gold Parquet outputs.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Most popular movies")
        st.dataframe(popular, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Highest-rated movies")
        st.dataframe(top_rated, use_container_width=True, hide_index=True)
    if genre is not None:
        st.subheader("Genre popularity")
        st.plotly_chart(px.bar(genre.head(15), x="rating_count", y="genre", orientation="h", title="Ratings by genre"), use_container_width=True)
        st.dataframe(genre, use_container_width=True, hide_index=True)
    if distribution is not None:
        st.subheader("Rating distribution")
        st.plotly_chart(px.bar(distribution, x="rating", y="percentage", title="Share of ratings by score", labels={"percentage": "Percentage (%)"}), use_container_width=True)
    if over_time is not None:
        st.subheader("Ratings over time")
        st.plotly_chart(px.line(over_time, x="period", y="rating_count", title="Ratings per month"), use_container_width=True)


def recommendations_page() -> None:
    page_header("03 / Recommendations", "A personal shortlist", "Stored ALS recommendations are shown first; unknown or low-history users use the documented popularity fallback.")
    recommendations = cached_recommendations()
    user_id = st.number_input("User ID", min_value=1, value=1, step=1)
    if recommendations is None:
        missing_artifact("Recommendation output", "Train ALS and generate recommendations before opening this page.")
        return
    selected = filter_recommendations(recommendations, int(user_id))
    if selected.empty:
        st.info("No stored recommendations exist for this user. Generate a batch that includes this user or use the recommendation module.")
        return
    source = selected["recommendation_source"].iloc[0] if "recommendation_source" in selected else "stored"
    st.success(f"Recommendation source: {source}")
    st.caption("ALS scores are predicted ratings. Popularity fallback scores are observed average ratings from Gold movie_stats.")
    st.dataframe(selected, use_container_width=True, hide_index=True)


def movie_explorer_page() -> None:
    page_header("04 / Movie Explorer", "Find a movie in the catalog", "Metadata comes from Gold movie_stats; unrated movies remain visible with null rating averages.")
    movies = cached_table("movie_stats")
    if movies is None:
        missing_artifact("Gold movie_stats", "Run the analytics stage to generate the movie catalog.")
        return
    query = st.text_input("Search title")
    genre = st.selectbox("Genre", ["All"] + available_genres(movies))
    filtered = filter_movies(movies, query, genre)
    st.caption(f"{len(filtered):,} movies match the current filters.")
    st.dataframe(filtered, use_container_width=True, hide_index=True)


def model_performance_page() -> None:
    page_header("05 / Model Performance", "How the ALS model performed", "Held-out rating prediction metrics from the saved model run.")
    metrics = cached_metrics()
    if metrics is None:
        missing_artifact("Model metrics", "Train ALS to generate `data/models/model_metrics.json`.")
        return
    values = metrics.get("metrics", {})
    rows = metrics.get("row_counts", {})
    metric_row([
        ("RMSE", format_number(values.get("rmse")), "Root mean squared error on held-out ratings."),
        ("MAE", format_number(values.get("mae")), "Mean absolute error on held-out ratings."),
        ("Training rows", format_number(rows.get("training")), None),
        ("Test rows", format_number(rows.get("test")), None),
    ])
    st.metric("Evaluated predictions", format_number(rows.get("predictions_evaluated")))
    st.info("RMSE and MAE measure rating prediction error. They do not independently establish ranking quality such as Precision@K or Recall@K; those metrics were not generated in this pipeline.")
    st.json(metrics.get("parameters", {}))


def pipeline_page() -> None:
    page_header("06 / Pipeline & Quality", "Artifact health", "Actual ingestion, cleaning, Gold, model, and recommendation artifacts available on this machine.")
    availability = artifact_availability()
    st.dataframe(pd.DataFrame([{"artifact": key, "available": value} for key, value in availability.items()]), use_container_width=True, hide_index=True)
    ingestion = load_ingestion_stats()
    quality = load_quality_stats()
    if ingestion is not None:
        st.subheader("Bronze ingestion")
        st.json(ingestion)
    else:
        missing_artifact("Bronze ingestion stats", "Run ingestion to generate the Bronze stats JSON.")
    if quality is not None:
        st.subheader("Silver data quality")
        st.json(quality)
    else:
        missing_artifact("Silver quality stats", "Run cleaning to generate the Silver stats JSON.")
    summary = cached_summary()
    if summary is not None:
        st.subheader("Gold table row counts")
        st.dataframe(pd.DataFrame([{"table": key, "rows": value} for key, value in summary.get("row_counts", {}).items()]), use_container_width=True, hide_index=True)


def main() -> None:
    st.sidebar.title("MovieLens Studio")
    st.sidebar.caption("Big Data Analytics project")
    page = st.sidebar.radio("Navigate", [
        "Overview", "Movie Analytics", "Personalized Recommendations",
        "Movie Explorer", "Model Performance", "Pipeline & Data Quality",
    ])
    try:
        {
            "Overview": overview_page,
            "Movie Analytics": movie_analytics_page,
            "Personalized Recommendations": recommendations_page,
            "Movie Explorer": movie_explorer_page,
            "Model Performance": model_performance_page,
            "Pipeline & Data Quality": pipeline_page,
        }[page]()
    except DashboardDataError as error:
        st.error(str(error))


if __name__ == "__main__":
    main()