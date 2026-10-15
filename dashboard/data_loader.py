"""Cached-friendly loaders and filters for dashboard artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recsys.config import configured_path, load_config  # noqa: E402


class DashboardDataError(ValueError):
    """Raised when an existing dashboard artifact cannot be parsed."""


def dashboard_config() -> dict[str, Any]:
    return load_config()


def artifact_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    configuration = config or dashboard_config()
    return {
        "gold": configured_path(configuration, "gold"),
        "models": configured_path(configuration, "models"),
        "recommendations": configured_path(configuration, "recommendations"),
        "bronze": configured_path(configuration, "bronze"),
        "silver": configured_path(configuration, "silver"),
    }


def load_json_artifact(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact, returning None only when it is unavailable."""
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as artifact_file:
            value = json.load(artifact_file)
    except (OSError, json.JSONDecodeError) as error:
        raise DashboardDataError(f"Could not read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise DashboardDataError(f"JSON artifact must contain an object: {path}")
    return value


def load_analytics_summary(config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return load_json_artifact(artifact_paths(config)["gold"] / "analytics_summary.json")


def load_model_metrics(config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return load_json_artifact(artifact_paths(config)["models"] / "model_metrics.json")


def load_ingestion_stats(config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return load_json_artifact(artifact_paths(config)["bronze"] / "ingestion_stats.json")


def load_quality_stats(config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    return load_json_artifact(artifact_paths(config)["silver"] / "data_quality_stats.json")


def load_gold_table(
    table_name: str, config: dict[str, Any] | None = None
) -> pd.DataFrame | None:
    path = artifact_paths(config)["gold"] / table_name
    if not path.is_dir():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as error:
        raise DashboardDataError(f"Could not read Gold table: {path}") from error


def load_recommendations(config: dict[str, Any] | None = None) -> pd.DataFrame | None:
    path = artifact_paths(config)["recommendations"] / "recommendations"
    if not path.is_dir():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as error:
        raise DashboardDataError(f"Could not read recommendation output: {path}") from error


def filter_movies(
    movies: pd.DataFrame | None,
    title_query: str = "",
    genre: str = "All",
) -> pd.DataFrame:
    if movies is None:
        return pd.DataFrame()
    result = movies.copy()
    if title_query.strip():
        result = result[
            result["title"].fillna("").str.contains(title_query.strip(), case=False, regex=False)
        ]
    if genre != "All" and "genres" in result.columns:
        result = result[result["genres"].apply(lambda values: genre in normalize_genres(values))]
    return result


def normalize_genres(value: object) -> list[str]:
    """Normalize Parquet list/array cells for safe pandas filtering."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [str(genre) for genre in value]


def available_genres(movies: pd.DataFrame | None) -> list[str]:
    if movies is None or "genres" not in movies.columns:
        return []
    values = {
        genre
        for genres in movies["genres"].dropna()
        for genre in normalize_genres(genres)
    }
    return sorted(values)


def filter_recommendations(
    recommendations: pd.DataFrame | None, user_id: int
) -> pd.DataFrame:
    if recommendations is None or "user_id" not in recommendations.columns:
        return pd.DataFrame()
    result = recommendations[recommendations["user_id"] == int(user_id)].copy()
    if "recommendation_rank" in result.columns:
        result = result.sort_values("recommendation_rank")
    return result


def artifact_availability(config: dict[str, Any] | None = None) -> dict[str, bool]:
    paths = artifact_paths(config)
    return {
        "analytics_summary": (paths["gold"] / "analytics_summary.json").is_file(),
        "model_metrics": (paths["models"] / "model_metrics.json").is_file(),
        "als_model": (paths["models"] / "als").is_dir(),
        "recommendations": (paths["recommendations"] / "recommendations").is_dir(),
        "ingestion_stats": (paths["bronze"] / "ingestion_stats.json").is_file(),
        "quality_stats": (paths["silver"] / "data_quality_stats.json").is_file(),
    }