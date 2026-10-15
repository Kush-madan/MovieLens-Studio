"""Generate ALS recommendations with a Spark popularity fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pyspark.ml.recommendation import ALSModel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .config import configured_path
from .model import load_als_model


def _read_required_parquet(
    spark: SparkSession, path: Path, description: str
) -> DataFrame:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    frame = spark.read.parquet(str(path))
    if frame.limit(1).count() == 0:
        raise ValueError(f"{description} is empty: {path}")
    return frame


def _target_users(
    spark: SparkSession,
    ratings: DataFrame,
    user_ids: Iterable[int] | None,
) -> DataFrame:
    if user_ids is None:
        return ratings.select("user_id").distinct()
    normalized_ids = list(dict.fromkeys(int(user_id) for user_id in user_ids))
    if not normalized_ids:
        raise ValueError("user_ids must contain at least one user")
    return spark.createDataFrame([(user_id,) for user_id in normalized_ids], ["user_id"])


def _rank_and_limit(frame: DataFrame, top_n: int) -> DataFrame:
    window = Window.partitionBy("user_id").orderBy(F.col("score").desc(), "movie_id")
    return frame.withColumn("recommendation_rank", F.row_number().over(window)).filter(
        F.col("recommendation_rank") <= top_n
    )


def _als_recommendations(
    model: ALSModel,
    users: DataFrame,
    ratings: DataFrame,
    movies: DataFrame,
    top_n: int,
) -> DataFrame:
    candidate_count = max(top_n * 5, top_n + 50)
    raw = model.recommendForUserSubset(users.select("user_id"), candidate_count)
    exploded = raw.select(
        "user_id", F.explode("recommendations").alias("recommendation")
    ).select(
        "user_id",
        F.col("recommendation.movie_id").cast("int").alias("movie_id"),
        F.col("recommendation.rating").cast("double").alias("score"),
    )
    unseen = exploded.join(
        ratings.select("user_id", "movie_id").dropDuplicates(),
        on=["user_id", "movie_id"],
        how="left_anti",
    )
    return _rank_and_limit(
        unseen.join(movies.select("movie_id", "title", "genres"), "movie_id", "inner")
        .withColumn("recommendation_source", F.lit("als")),
        top_n,
    )


def _popularity_recommendations(
    users: DataFrame,
    ratings: DataFrame,
    movie_stats: DataFrame,
    top_n: int,
) -> DataFrame:
    candidates = movie_stats.filter(F.col("rating_count") > 0).select(
        "movie_id", "title", "genres", "rating_count", "avg_rating"
    )
    candidate_pairs = users.crossJoin(candidates).join(
        ratings.select("user_id", "movie_id").dropDuplicates(),
        on=["user_id", "movie_id"],
        how="left_anti",
    ).select(
        "user_id",
        "movie_id",
        "title",
        "genres",
        F.col("avg_rating").cast("double").alias("score"),
    ).withColumn("recommendation_source", F.lit("popularity_fallback"))
    return _rank_and_limit(candidate_pairs, top_n)


def generate_recommendations(
    spark: SparkSession,
    config: dict[str, Any],
    user_ids: Iterable[int] | None = None,
    top_n: int | None = None,
) -> DataFrame:
    """Generate ALS recommendations and fallback results, writing Parquet output."""
    recommendation_config = config.get("recommendations", {})
    if not isinstance(recommendation_config, dict):
        raise ValueError("The recommendations configuration must be a YAML mapping")
    requested_top_n = int(
        top_n if top_n is not None else recommendation_config.get("top_n", 10)
    )
    min_user_ratings = int(recommendation_config.get("min_user_ratings", 5))
    if requested_top_n <= 0 or min_user_ratings < 0:
        raise ValueError("top_n must be positive and min_user_ratings cannot be negative")

    silver_path = configured_path(config, "silver")
    ratings = _read_required_parquet(spark, silver_path / "ratings", "Silver ratings")
    movies = _read_required_parquet(spark, silver_path / "movies", "Silver movies")
    movie_stats = _read_required_parquet(
        spark, configured_path(config, "gold") / "movie_stats", "Gold movie_stats"
    )
    target_users = _target_users(spark, ratings, user_ids).cache()
    rating_counts = ratings.groupBy("user_id").agg(F.count("*").alias("history_count"))
    als_users = target_users.join(rating_counts, "user_id", "left").filter(
        F.col("history_count") >= min_user_ratings
    ).select("user_id")
    fallback_users = target_users.join(als_users, "user_id", "left_anti")

    fallback_results = _popularity_recommendations(
        fallback_users, ratings, movie_stats, requested_top_n
    )
    if als_users.limit(1).count() > 0:
        als_model = load_als_model(config)
        als_results = _als_recommendations(
            als_model, als_users, ratings, movies, requested_top_n
        )
    else:
        als_results = fallback_results.limit(0)
    results = als_results.unionByName(fallback_results).select(
        "user_id",
        "movie_id",
        "title",
        "genres",
        "score",
        "recommendation_source",
        "recommendation_rank",
    )
    output_path = configured_path(config, "recommendations") / "recommendations"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.write.mode("overwrite").parquet(str(output_path))
    target_users.unpersist()
    return results