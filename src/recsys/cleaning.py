"""Clean Bronze MovieLens data into validated Silver Parquet datasets."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .config import configured_path


LOGGER = logging.getLogger(__name__)


def _read_bronze(spark: SparkSession, bronze_path: Path, table: str) -> DataFrame:
    path = bronze_path / table
    if not path.is_dir():
        raise FileNotFoundError(f"Bronze dataset does not exist: {path}")
    return spark.read.parquet(str(path))


def _null_counts(frame: DataFrame, columns: list[str]) -> dict[str, int]:
    result = frame.select(
        *[F.sum(F.col(column).isNull().cast("int")).alias(column) for column in columns]
    ).first()
    return {column: int(result[column] or 0) for column in columns}


def _write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2, sort_keys=True)


def _clean_movies(frame: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    rows_in = frame.count()
    raw_columns = ["movie_id", "title", "genres"]
    null_counts = _null_counts(frame, raw_columns)
    typed = frame.select(
        F.col("movie_id").cast("int").alias("movie_id"),
        F.trim(F.col("title")).alias("title"),
        F.trim(F.col("genres")).alias("genres_raw"),
        F.col("ingested_at").cast("timestamp").alias("ingested_at"),
        F.col("source_file").cast("string").alias("source_file"),
    )
    valid = typed.filter(
        F.col("movie_id").isNotNull()
        & (F.col("movie_id") > 0)
        & F.col("title").isNotNull()
        & F.col("genres_raw").isNotNull()
    )
    exact_duplicates = valid.count() - valid.dropDuplicates().count()
    cleaned = (
        valid.dropDuplicates()
        .withColumn(
            "release_year",
            F.regexp_extract(F.col("title"), r"\((\d{4})\)\s*$", 1).cast("int"),
        )
        .withColumn(
            "genres",
            F.when(
                F.col("genres_raw") == "(no genres listed)",
                F.array().cast("array<string>"),
            ).otherwise(F.split(F.col("genres_raw"), "\\|")),
        )
        .select(
            "movie_id",
            "title",
            "release_year",
            "genres",
            "ingested_at",
            "source_file",
        )
    )
    return cleaned, {
        "rows_in": rows_in,
        "rows_out": cleaned.count(),
        "null_counts": null_counts,
        "exact_duplicate_rows": int(exact_duplicates),
    }


def _clean_users(frame: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    rows_in = frame.count()
    raw_columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    null_counts = _null_counts(frame, raw_columns)
    typed = frame.select(
        F.col("user_id").cast("int").alias("user_id"),
        F.trim(F.col("gender")).alias("gender"),
        F.col("age").cast("int").alias("age"),
        F.col("occupation").cast("int").alias("occupation"),
        F.trim(F.col("zip_code")).alias("zip_code"),
        F.col("ingested_at").cast("timestamp").alias("ingested_at"),
        F.col("source_file").cast("string").alias("source_file"),
    )
    valid = typed.filter(F.col("user_id").isNotNull() & (F.col("user_id") > 0))
    exact_duplicates = valid.count() - valid.dropDuplicates().count()
    cleaned = valid.dropDuplicates()
    return cleaned, {
        "rows_in": rows_in,
        "rows_out": cleaned.count(),
        "null_counts": null_counts,
        "exact_duplicate_rows": int(exact_duplicates),
    }


def _clean_ratings(
    frame: DataFrame,
    movies: DataFrame,
    rating_min: float,
    rating_max: float,
) -> tuple[DataFrame, dict[str, Any]]:
    rows_in = frame.count()
    raw_columns = ["user_id", "movie_id", "rating", "timestamp"]
    null_counts = _null_counts(frame, raw_columns)
    typed = frame.select(
        F.col("user_id").cast("int").alias("user_id"),
        F.col("movie_id").cast("int").alias("movie_id"),
        F.col("rating").cast("double").alias("rating"),
        F.to_timestamp(F.from_unixtime(F.col("timestamp").cast("long"))).alias(
            "timestamp"
        ),
        F.col("ingested_at").cast("timestamp").alias("ingested_at"),
        F.col("source_file").cast("string").alias("source_file"),
    )
    now = F.current_timestamp()
    rejected_counts = {
        "null_required_fields": typed.filter(
            F.col("user_id").isNull()
            | F.col("movie_id").isNull()
            | F.col("rating").isNull()
            | F.col("timestamp").isNull()
        ).count(),
        "user_id_non_positive": typed.filter(
            F.col("user_id").isNotNull() & (F.col("user_id") <= 0)
        ).count(),
        "movie_id_non_positive": typed.filter(
            F.col("movie_id").isNotNull() & (F.col("movie_id") <= 0)
        ).count(),
        "rating_out_of_range": typed.filter(
            F.col("rating").isNotNull()
            & ((F.col("rating") < rating_min) | (F.col("rating") > rating_max))
        ).count(),
        "timestamp_in_future": typed.filter(
            F.col("timestamp").isNotNull() & (F.col("timestamp") > now)
        ).count(),
    }
    valid = typed.filter(
        F.col("user_id").isNotNull()
        & (F.col("user_id") > 0)
        & F.col("movie_id").isNotNull()
        & (F.col("movie_id") > 0)
        & F.col("rating").isNotNull()
        & (F.col("rating") >= rating_min)
        & (F.col("rating") <= rating_max)
        & F.col("timestamp").isNotNull()
        & (F.col("timestamp") <= now)
    )
    exact_duplicate_rows = valid.count() - valid.dropDuplicates().count()
    deduplicated = valid.dropDuplicates()
    pair_duplicate_rows = deduplicated.count() - deduplicated.select(
        "user_id", "movie_id"
    ).dropDuplicates().count()
    latest_window = Window.partitionBy("user_id", "movie_id").orderBy(
        F.col("timestamp").desc(), F.col("ingested_at").desc()
    )
    latest = deduplicated.withColumn("pair_rank", F.row_number().over(latest_window)).filter(
        F.col("pair_rank") == 1
    ).drop("pair_rank")
    orphan_count = latest.join(
        movies.select("movie_id").dropDuplicates(), on="movie_id", how="left_anti"
    ).count()
    cleaned = latest.join(
        movies.select("movie_id").dropDuplicates(), on="movie_id", how="left_semi"
    )
    rejected_counts["orphan_movie_id"] = orphan_count
    return cleaned, {
        "rows_in": rows_in,
        "rows_out": cleaned.count(),
        "null_counts": null_counts,
        "rejected_counts": rejected_counts,
        "exact_duplicate_rows": int(exact_duplicate_rows),
        "duplicate_user_movie_rows": int(pair_duplicate_rows),
        "orphan_rows": int(orphan_count),
    }


def clean_all(spark: SparkSession, config: dict[str, Any]) -> dict[str, Any]:
    """Clean all Bronze datasets into Silver and write quality statistics."""
    bronze_path = configured_path(config, "bronze")
    silver_path = configured_path(config, "silver")
    silver_path.mkdir(parents=True, exist_ok=True)
    cleaning_config = config.get("cleaning", {})
    if not isinstance(cleaning_config, dict):
        raise ValueError("The cleaning configuration must be a YAML mapping")
    rating_min = float(cleaning_config.get("rating_min", 0.5))
    rating_max = float(cleaning_config.get("rating_max", 5.0))
    if rating_min > rating_max:
        raise ValueError("cleaning.rating_min cannot exceed cleaning.rating_max")

    movies, movie_stats = _clean_movies(_read_bronze(spark, bronze_path, "movies"))
    users, user_stats = _clean_users(_read_bronze(spark, bronze_path, "users"))
    ratings, rating_stats = _clean_ratings(
        _read_bronze(spark, bronze_path, "ratings"),
        movies,
        rating_min,
        rating_max,
    )

    movies.write.mode("overwrite").parquet(str(silver_path / "movies"))
    users.write.mode("overwrite").parquet(str(silver_path / "users"))
    ratings.write.mode("overwrite").parquet(str(silver_path / "ratings"))

    stats = {
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "rating_bounds": {"min": rating_min, "max": rating_max},
        "tables": {"movies": movie_stats, "users": user_stats, "ratings": rating_stats},
        "rows_in": rating_stats["rows_in"],
        "rows_out": rating_stats["rows_out"],
        "per_rule_rejected_counts": rating_stats["rejected_counts"],
        "duplicate_count": rating_stats["exact_duplicate_rows"]
        + rating_stats["duplicate_user_movie_rows"],
        "orphan_count": rating_stats["orphan_rows"],
        "null_counts": {
            "movies": movie_stats["null_counts"],
            "users": user_stats["null_counts"],
            "ratings": rating_stats["null_counts"],
        },
    }
    _write_stats(silver_path / "data_quality_stats.json", stats)
    LOGGER.info(
        "Silver cleaning completed: %s orphan ratings removed",
        rating_stats["orphan_rows"],
    )
    return stats