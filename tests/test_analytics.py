"""Tests for Gold analytics."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from recsys.analytics import analytics_all


def _config(tmp_path: Path) -> dict:
    return {
        "analytics": {
            "top_n": 1,
            "min_ratings": 2,
            "activity_buckets": [{"name": "1+", "min": 1}],
        },
        "paths": {"silver": str(tmp_path / "silver"), "gold": str(tmp_path / "gold")},
    }


def _write_silver(spark, tmp_path: Path) -> None:
    silver_path = tmp_path / "silver"
    timestamp = datetime(2024, 1, 1, 12, 0, 0)
    movies = spark.createDataFrame(
        [
            (1, "Movie One (2000)", 2000, ["Comedy", "Drama"]),
            (2, "Movie Two (2001)", 2001, ["Drama"]),
            (3, "Movie Three (2002)", 2002, ["Comedy"]),
            (4, "Unrated Movie (2003)", 2003, ["Action"]),
        ],
        ["movie_id", "title", "release_year", "genres"],
    )
    ratings = spark.createDataFrame(
        [
            (1, 1, 5.0, timestamp),
            (2, 1, 4.0, timestamp),
            (1, 2, 2.0, timestamp),
            (2, 3, 3.0, timestamp),
        ],
        ["user_id", "movie_id", "rating", "timestamp"],
    )
    users = spark.createDataFrame(
        [(1, "M", 25), (2, "F", 35)], ["user_id", "gender", "age"]
    )
    movies.write.mode("overwrite").parquet(str(silver_path / "movies"))
    ratings.write.mode("overwrite").parquet(str(silver_path / "ratings"))
    users.write.mode("overwrite").parquet(str(silver_path / "users"))


def test_analytics_outputs_respect_thresholds_and_create_files(spark, tmp_path):
    _write_silver(spark, tmp_path)
    summary = analytics_all(spark, _config(tmp_path))
    gold_path = tmp_path / "gold"

    assert summary["row_counts"]["most_popular_movies"] == 1
    assert summary["row_counts"]["movie_stats"] == 4
    assert summary["sparsity"] == 0.5
    assert summary["density"] == 0.5
    movie_stats = spark.read.parquet(str(gold_path / "movie_stats"))
    unrated = movie_stats.filter("movie_id = 4").first()
    assert unrated.rating_count == 0
    assert unrated.avg_rating is None
    assert unrated.rating_stddev is None
    top_rated = spark.read.parquet(str(gold_path / "top_rated_movies"))
    assert top_rated.select("movie_id").collect()[0][0] == 1
    assert (gold_path / "analytics_summary.json").is_file()
    for name in summary["row_counts"]:
        assert (gold_path / name).is_dir()


def test_genre_explode_and_rating_distribution_percentages(spark, tmp_path):
    _write_silver(spark, tmp_path)
    analytics_all(spark, _config(tmp_path))
    gold_path = tmp_path / "gold"

    genre_rows = {
        row.genre: row.rating_count
        for row in spark.read.parquet(str(gold_path / "genre_popularity")).collect()
    }
    assert genre_rows["Comedy"] == 3
    assert genre_rows["Drama"] == 3

    distribution = spark.read.parquet(str(gold_path / "rating_distribution"))
    assert abs(distribution.agg({"percentage": "sum"}).first()[0] - 100.0) < 1e-9


def test_missing_required_silver_dataset_raises(spark, tmp_path):
    (tmp_path / "silver").mkdir()
    try:
        analytics_all(spark, _config(tmp_path))
    except FileNotFoundError as error:
        assert "movies" in str(error)
    else:
        raise AssertionError("Missing Silver movies dataset did not raise")