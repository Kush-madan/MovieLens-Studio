"""Tests for dashboard artifact loading and pure filtering behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.data_loader import (
    artifact_availability,
    available_genres,
    filter_movies,
    filter_recommendations,
    load_json_artifact,
)


def test_json_loading_and_missing_artifact(tmp_path: Path):
    artifact = tmp_path / "analytics_summary.json"
    artifact.write_text(json.dumps({"total_movies": 3883}), encoding="utf-8")

    assert load_json_artifact(artifact)["total_movies"] == 3883
    assert load_json_artifact(tmp_path / "missing.json") is None


def test_movie_search_genre_filter_and_null_average():
    movies = pd.DataFrame(
        [
            {"title": "Café Society", "genres": ["Comedy"], "avg_rating": 4.0},
            {"title": "Unrated Film", "genres": ["Drama"], "avg_rating": None},
        ]
    )

    result = filter_movies(movies, "café", "Comedy")
    assert result.iloc[0]["title"] == "Café Society"
    assert pd.isna(filter_movies(movies, genre="Drama").iloc[0]["avg_rating"])
    assert available_genres(movies) == ["Comedy", "Drama"]


def test_genre_helpers_handle_parquet_array_values():
    movies = pd.DataFrame(
        [{"title": "Array Genres", "genres": np.array(["Action", "Drama"])}]
    )

    assert available_genres(movies) == ["Action", "Drama"]
    assert len(filter_movies(movies, genre="Drama")) == 1


def test_recommendations_filter_by_user_and_fallback_source():
    recommendations = pd.DataFrame(
        [
            {"user_id": 7, "title": "Movie A", "recommendation_source": "als", "recommendation_rank": 1},
            {"user_id": 8, "title": "Movie B", "recommendation_source": "popularity_fallback", "recommendation_rank": 1},
        ]
    )

    assert filter_recommendations(recommendations, 7).iloc[0]["title"] == "Movie A"
    assert filter_recommendations(recommendations, 8).iloc[0]["recommendation_source"] == "popularity_fallback"
    assert filter_recommendations(recommendations, 99).empty


def test_artifact_availability_reports_missing_files(tmp_path: Path):
    config = {
        "paths": {
            "gold": str(tmp_path / "gold"),
            "models": str(tmp_path / "models"),
            "recommendations": str(tmp_path / "recommendations"),
            "bronze": str(tmp_path / "bronze"),
            "silver": str(tmp_path / "silver"),
        }
    }

    availability = artifact_availability(config)
    assert not any(availability.values())