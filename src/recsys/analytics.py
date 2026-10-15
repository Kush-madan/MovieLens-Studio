"""Compute Gold analytics from the Silver MovieLens Parquet datasets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .config import configured_path


class SilverDatasetNotFoundError(FileNotFoundError):
    """Raised when a required Silver dataset is missing."""


class EmptySilverDatasetError(ValueError):
    """Raised when a required Silver dataset has no rows."""


def _read_silver(
    spark: SparkSession,
    silver_path: Path,
    table: str,
    required: bool = True,
) -> DataFrame | None:
    path = silver_path / table
    if not path.is_dir():
        if required:
            raise SilverDatasetNotFoundError(f"Silver dataset does not exist: {path}")
        return None
    frame = spark.read.parquet(str(path))
    if frame.limit(1).count() == 0:
        if required:
            raise EmptySilverDatasetError(f"Silver dataset is empty: {path}")
        return None
    return frame


def _write_dataset(frame: DataFrame, gold_path: Path, name: str) -> int:
    output_path = gold_path / name
    frame.write.mode("overwrite").parquet(str(output_path))
    return frame.count()


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, sort_keys=True)


def _activity_bucket_column(
    rating_count_column: F.Column, bucket_config: list[dict[str, Any]]
) -> F.Column:
    bucket_column: F.Column | None = None
    for bucket in bucket_config:
        name = str(bucket["name"])
        minimum = int(bucket["min"])
        condition = rating_count_column >= minimum
        if bucket.get("max") is not None:
            condition = condition & (rating_count_column <= int(bucket["max"]))
        bucket_column = F.when(condition, F.lit(name)) if bucket_column is None else bucket_column.when(condition, F.lit(name))
    if bucket_column is None:
        raise ValueError("analytics.activity_buckets must contain at least one bucket")
    return bucket_column.otherwise(F.lit("other"))


def analytics_all(spark: SparkSession, config: dict[str, Any]) -> dict[str, Any]:
    """Compute all Gold analytics and return the written summary."""
    silver_path = configured_path(config, "silver")
    gold_path = configured_path(config, "gold")
    gold_path.mkdir(parents=True, exist_ok=True)
    analytics_config = config.get("analytics", {})
    if not isinstance(analytics_config, dict):
        raise ValueError("The analytics configuration must be a YAML mapping")
    top_n = int(analytics_config.get("top_n", 10))
    min_ratings = int(analytics_config.get("min_ratings", 20))
    if top_n <= 0:
        raise ValueError("analytics.top_n must be greater than zero")
    if min_ratings < 0:
        raise ValueError("analytics.min_ratings cannot be negative")
    bucket_config = analytics_config.get("activity_buckets", [])
    if not isinstance(bucket_config, list):
        raise ValueError("analytics.activity_buckets must be a list")

    movies = _read_silver(spark, silver_path, "movies")
    ratings = _read_silver(spark, silver_path, "ratings")
    users = _read_silver(spark, silver_path, "users", required=False)
    assert movies is not None
    assert ratings is not None
    movies.cache()
    ratings.cache()
    ratings_count = ratings.count()
    movies_count = movies.count()
    users_count = users.count() if users is not None else 0
    rated_movies = ratings.join(movies, on="movie_id", how="inner")

    rating_stats_by_movie = ratings.groupBy("movie_id").agg(
        F.count("rating").alias("rating_count"),
        F.avg("rating").alias("avg_rating"),
        F.stddev_samp("rating").alias("rating_stddev"),
    )
    movie_stats = movies.select(
        "movie_id", "title", "genres", F.col("release_year").alias("year")
    ).join(rating_stats_by_movie, on="movie_id", how="left").withColumn(
        "rating_count", F.coalesce(F.col("rating_count"), F.lit(0).cast("long"))
    )
    most_popular_movies = movie_stats.orderBy(
        F.col("rating_count").desc(), F.col("movie_id").asc()
    ).limit(top_n)
    top_rated_movies = movie_stats.filter(F.col("rating_count") >= min_ratings).orderBy(
        F.col("avg_rating").desc(), F.col("rating_count").desc(), F.col("movie_id").asc()
    ).limit(top_n)

    genre_ratings = rated_movies.select(
        "movie_id", "rating", F.explode("genres").alias("genre")
    ).filter(F.col("genre").isNotNull() & (F.col("genre") != ""))
    genre_popularity = genre_ratings.groupBy("genre").agg(
        F.count("rating").alias("rating_count"),
        F.avg("rating").alias("avg_rating"),
        F.countDistinct("movie_id").alias("movie_count"),
    ).orderBy(F.col("rating_count").desc(), F.col("genre").asc())

    genre_stats_by_decade = genre_ratings.join(
        movies.select("movie_id", "release_year"), on="movie_id", how="inner"
    ).withColumn(
        "decade",
        (F.floor(F.col("release_year") / F.lit(10)) * F.lit(10)).cast("int"),
    ).filter(F.col("decade").isNotNull()).groupBy("decade", "genre").agg(
        F.count("rating").alias("rating_count"),
        F.avg("rating").alias("avg_rating"),
        F.countDistinct("movie_id").alias("movie_count"),
    ).orderBy("decade", F.col("rating_count").desc(), "genre")

    distribution_base = ratings.groupBy("rating").agg(F.count("*").alias("rating_count"))
    rating_distribution = distribution_base.withColumn(
        "percentage",
        F.col("rating_count") / F.lit(ratings_count) * F.lit(100.0),
    ).orderBy("rating")

    user_activity = ratings.groupBy("user_id").agg(
        F.count("rating").alias("rating_count"),
        F.avg("rating").alias("avg_rating"),
        F.min("timestamp").alias("first_rating_at"),
        F.max("timestamp").alias("last_rating_at"),
    )
    user_activity_buckets = user_activity.withColumn(
        "activity_bucket",
        _activity_bucket_column(F.col("rating_count"), bucket_config),
    ).groupBy("activity_bucket").agg(
        F.count("*").alias("user_count"),
        F.avg("rating_count").alias("avg_rating_count"),
    ).orderBy("activity_bucket")

    ratings_over_time = ratings.withColumn(
        "period", F.date_format(F.col("timestamp"), "yyyy-MM")
    ).groupBy("period").agg(
        F.count("*").alias("rating_count"),
        F.avg("rating").alias("avg_rating"),
    ).orderBy("period")

    output_frames: dict[str, DataFrame] = {
        "movie_stats": movie_stats,
        "most_popular_movies": most_popular_movies,
        "top_rated_movies": top_rated_movies,
        "genre_popularity": genre_popularity,
        "genre_stats_by_decade": genre_stats_by_decade,
        "rating_distribution": rating_distribution,
        "user_activity": user_activity,
        "user_activity_buckets": user_activity_buckets,
        "ratings_over_time": ratings_over_time,
    }

    if users is not None:
        demographics = ratings.join(users, on="user_id", how="inner").withColumn(
            "age_group",
            F.when(F.col("age") < 18, "under_18")
            .when(F.col("age") < 25, "18-24")
            .when(F.col("age") < 35, "25-34")
            .when(F.col("age") < 45, "35-44")
            .when(F.col("age") < 56, "45-55")
            .otherwise("56+"),
        ).groupBy("gender", "age_group").agg(
            F.avg("rating").alias("avg_rating"),
            F.count("rating").alias("rating_count"),
            F.countDistinct("user_id").alias("user_count"),
        ).orderBy("gender", "age_group")
        output_frames["demographics"] = demographics

    row_counts = {
        name: _write_dataset(frame, gold_path, name)
        for name, frame in output_frames.items()
    }
    global_stats = ratings.agg(
        F.avg("rating").alias("global_average_rating"),
        F.min("timestamp").alias("date_start"),
        F.max("timestamp").alias("date_end"),
    ).first()
    possible_ratings = users_count * movies_count
    density = (ratings_count / possible_ratings) if possible_ratings else 0.0
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_users": users_count,
        "total_movies": movies_count,
        "total_ratings": ratings_count,
        "global_average_rating": float(global_stats["global_average_rating"]),
        "sparsity": 1.0 - density,
        "density": density,
        "date_range": {
            "start": global_stats["date_start"].isoformat(),
            "end": global_stats["date_end"].isoformat(),
        },
        "row_counts": row_counts,
        "top_n": top_n,
        "min_ratings": min_ratings,
    }
    _write_summary(gold_path / "analytics_summary.json", summary)
    movies.unpersist()
    ratings.unpersist()
    return summary