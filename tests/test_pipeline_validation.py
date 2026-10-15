"""Tests for Airflow-facing pipeline output validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recsys.pipeline_validation import PipelineValidationError, validate_pipeline_outputs


def _config(tmp_path: Path) -> dict:
    return {
        "validation": {"expected_users": 1, "expected_movies": 1, "expected_ratings": 1},
        "paths": {
            "bronze": str(tmp_path / "bronze"),
            "silver": str(tmp_path / "silver"),
            "gold": str(tmp_path / "gold"),
            "models": str(tmp_path / "models"),
            "recommendations": str(tmp_path / "recommendations"),
        },
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_valid_outputs(spark, tmp_path: Path) -> dict:
    config = _config(tmp_path)
    _write_json(
        tmp_path / "bronze" / "ingestion_stats.json",
        {"tables": {"ratings": {"row_count": 1}}},
    )
    _write_json(tmp_path / "silver" / "data_quality_stats.json", {"rows_out": 1})
    _write_json(
        tmp_path / "gold" / "analytics_summary.json",
        {
            "total_users": 1,
            "total_movies": 1,
            "total_ratings": 1,
            "density": 1.0,
            "sparsity": 0.0,
        },
    )
    _write_json(tmp_path / "models" / "model_metrics.json", {"metrics": {"rmse": 1.0, "mae": 0.5}})
    (tmp_path / "models" / "als").mkdir(parents=True)
    recommendations = spark.createDataFrame([(1, 1)], ["user_id", "movie_id"])
    recommendations.write.mode("overwrite").parquet(str(tmp_path / "recommendations" / "recommendations"))
    movie_stats = spark.createDataFrame([(1, "Movie")], ["movie_id", "title"])
    movie_stats.write.mode("overwrite").parquet(str(tmp_path / "gold" / "movie_stats"))
    return config


def test_pipeline_output_validation_passes_for_valid_fixture(spark, tmp_path):
    config = _write_valid_outputs(spark, tmp_path)

    report = validate_pipeline_outputs(spark, config)

    assert report["counts"] == {"users": 1, "movies": 1, "ratings": 1}
    assert report["movie_stats_rows"] == 1


def test_pipeline_output_validation_fails_for_missing_artifact(spark, tmp_path):
    config = _write_valid_outputs(spark, tmp_path)
    (tmp_path / "models" / "model_metrics.json").unlink()

    with pytest.raises(PipelineValidationError, match="model_metrics.json"):
        validate_pipeline_outputs(spark, config)