"""Tests for ALS recommendations and popularity fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from recsys.model import train_als
from recsys.recommend import generate_recommendations


def _config(tmp_path: Path) -> dict:
    return {
        "project": {"random_seed": 42},
        "model": {
            "rank": 2,
            "max_iter": 3,
            "reg_param": 0.1,
            "seed": 42,
            "train_fraction": 0.75,
        },
        "recommendations": {"top_n": 2, "min_user_ratings": 2},
        "paths": {
            "silver": str(tmp_path / "silver"),
            "models": str(tmp_path / "models"),
            "gold": str(tmp_path / "gold"),
            "recommendations": str(tmp_path / "recommendations"),
        },
    }


def _write_silver_and_gold(spark, tmp_path: Path) -> None:
    ratings = spark.createDataFrame(
        [
            (1, 1, 5.0), (1, 2, 4.0), (1, 3, 1.0),
            (2, 1, 4.0), (2, 2, 3.0), (2, 4, 5.0),
            (3, 1, 5.0), (3, 3, 2.0), (3, 4, 4.0),
            (4, 2, 5.0), (4, 3, 4.0), (4, 4, 3.0),
        ],
        ["user_id", "movie_id", "rating"],
    )
    movies = spark.createDataFrame(
        [
            (1, "Movie One", ["Drama"]),
            (2, "Movie Two", ["Comedy"]),
            (3, "Movie Three", ["Drama"]),
            (4, "Movie Four", ["Action"]),
            (5, "Unrated Candidate", ["Action"]),
        ],
        ["movie_id", "title", "genres"],
    )
    movie_stats = spark.createDataFrame(
        [
            (1, "Movie One", ["Drama"], 3, 4.67),
            (2, "Movie Two", ["Comedy"], 3, 4.0),
            (3, "Movie Three", ["Drama"], 3, 2.33),
            (4, "Movie Four", ["Action"], 3, 4.0),
            (5, "Unrated Candidate", ["Action"], 0, None),
        ],
        ["movie_id", "title", "genres", "rating_count", "avg_rating"],
    )
    ratings.write.mode("overwrite").parquet(str(tmp_path / "silver" / "ratings"))
    movies.write.mode("overwrite").parquet(str(tmp_path / "silver" / "movies"))
    movie_stats.write.mode("overwrite").parquet(str(tmp_path / "gold" / "movie_stats"))


def test_recommendations_exclude_rated_movies_join_metadata_and_fallback(
    spark, tmp_path
):
    config = _config(tmp_path)
    _write_silver_and_gold(spark, tmp_path)
    train_als(spark, config)

    results = generate_recommendations(spark, config, user_ids=[1, 99], top_n=2).collect()
    user_one = [row for row in results if row.user_id == 1]
    fallback = [row for row in results if row.user_id == 99]
    rated_by_user_one = {1, 2, 3}

    assert user_one
    assert all(row.movie_id not in rated_by_user_one for row in user_one)
    assert all(row.title and row.genres for row in user_one)
    assert fallback
    assert all(row.recommendation_source == "popularity_fallback" for row in fallback)
    assert all(row.title != "Unrated Candidate" for row in fallback)
    assert (tmp_path / "recommendations" / "recommendations").is_dir()


def test_missing_model_artifact_raises(spark, tmp_path):
    config = _config(tmp_path)
    _write_silver_and_gold(spark, tmp_path)

    with pytest.raises(FileNotFoundError, match="ALS model artifact"):
        generate_recommendations(spark, config, user_ids=[1])