"""Validate artifacts produced by the MovieLens pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

from .config import configured_path


class PipelineValidationError(ValueError):
    """Raised when a required pipeline artifact or invariant is invalid."""


def _required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineValidationError(f"Required JSON artifact is missing: {path}")
    try:
        with path.open(encoding="utf-8") as artifact_file:
            value = json.load(artifact_file)
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineValidationError(f"Could not read JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise PipelineValidationError(f"JSON artifact must contain an object: {path}")
    return value


def _required_value(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise PipelineValidationError(f"Missing field '{path}' in pipeline artifacts")
        value = value[part]
    return value


def validate_pipeline_outputs(
    spark: SparkSession, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate required outputs and MovieLens dataset invariants."""
    bronze = configured_path(config, "bronze")
    silver = configured_path(config, "silver")
    gold = configured_path(config, "gold")
    models = configured_path(config, "models")
    recommendations = configured_path(config, "recommendations") / "recommendations"

    ingestion = _required_json(bronze / "ingestion_stats.json")
    quality = _required_json(silver / "data_quality_stats.json")
    summary = _required_json(gold / "analytics_summary.json")
    metrics = _required_json(models / "model_metrics.json")
    model_path = models / "als"
    if not model_path.is_dir():
        raise PipelineValidationError(f"ALS model artifact is missing: {model_path}")
    if not recommendations.is_dir() or not list(recommendations.rglob("*.parquet")):
        raise PipelineValidationError(
            f"Recommendation Parquet output is missing or empty: {recommendations}"
        )

    validation_config = config.get("validation", {})
    if not isinstance(validation_config, dict):
        raise PipelineValidationError("The validation configuration must be a mapping")
    expected_users = int(validation_config.get("expected_users", 6040))
    expected_movies = int(validation_config.get("expected_movies", 3883))
    expected_ratings = int(validation_config.get("expected_ratings", 1000209))
    actual_counts = {
        "users": int(_required_value(summary, "total_users")),
        "movies": int(_required_value(summary, "total_movies")),
        "ratings": int(_required_value(summary, "total_ratings")),
    }
    expected_counts = {
        "users": expected_users,
        "movies": expected_movies,
        "ratings": expected_ratings,
    }
    if actual_counts != expected_counts:
        raise PipelineValidationError(
            f"Dataset counts do not match expectations: actual={actual_counts}, "
            f"expected={expected_counts}"
        )

    movie_stats_path = gold / "movie_stats"
    if not movie_stats_path.is_dir():
        raise PipelineValidationError(f"Gold movie_stats dataset is missing: {movie_stats_path}")
    movie_stats_count = spark.read.parquet(str(movie_stats_path)).count()
    if movie_stats_count != expected_movies:
        raise PipelineValidationError(
            f"Gold movie_stats contains {movie_stats_count} rows; expected {expected_movies}"
        )

    density = float(_required_value(summary, "density"))
    sparsity = float(_required_value(summary, "sparsity"))
    if not math.isfinite(density) or not math.isfinite(sparsity):
        raise PipelineValidationError("Density and sparsity must be finite numbers")
    if abs((density + sparsity) - 1.0) > 1e-9:
        raise PipelineValidationError(
            f"Density and sparsity must sum to one: density={density}, sparsity={sparsity}"
        )

    rmse = float(_required_value(metrics, "metrics.rmse"))
    mae = float(_required_value(metrics, "metrics.mae"))
    if not math.isfinite(rmse) or not math.isfinite(mae):
        raise PipelineValidationError("Model RMSE and MAE must be finite numbers")

    ingestion_ratings = int(_required_value(ingestion, "tables.ratings.row_count"))
    if ingestion_ratings != expected_ratings:
        raise PipelineValidationError(
            f"Bronze ratings count {ingestion_ratings} does not match expected {expected_ratings}"
        )
    return {
        "counts": actual_counts,
        "movie_stats_rows": movie_stats_count,
        "density": density,
        "sparsity": sparsity,
        "rmse": rmse,
        "mae": mae,
        "quality_rows_out": _required_value(quality, "rows_out"),
        "recommendation_output": str(recommendations),
    }