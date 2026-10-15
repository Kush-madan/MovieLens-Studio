"""Tests for ALS training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from recsys.model import (
    calculate_ranking_metrics,
    evaluate_predictions,
    load_als_model,
    train_als,
)


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
        "paths": {"silver": str(tmp_path / "silver"), "models": str(tmp_path / "models")},
    }


def _write_ratings(spark, tmp_path: Path) -> None:
    rows = [
        (1, 1, 5.0), (1, 2, 4.0), (1, 3, 1.0), (1, 4, 2.0),
        (2, 1, 4.0), (2, 2, 3.0), (2, 3, 2.0), (2, 4, 5.0),
        (3, 1, 5.0), (3, 2, 2.0), (3, 3, 4.0), (3, 4, 3.0),
        (4, 1, 3.0), (4, 2, 5.0), (4, 3, 2.0), (4, 4, 4.0),
    ]
    frame = spark.createDataFrame(rows, ["user_id", "movie_id", "rating"])
    frame.write.mode("overwrite").parquet(str(tmp_path / "silver" / "ratings"))


def test_evaluate_predictions_calculates_rmse_and_mae(spark):
    predictions = spark.createDataFrame(
        [(5.0, 4.0), (2.0, 3.0), (4.0, None)], ["rating", "prediction"]
    )
    metrics = evaluate_predictions(predictions)

    assert metrics["prediction_row_count"] == 2
    assert metrics["rmse"] == 1.0
    assert metrics["mae"] == 1.0


def test_calculate_ranking_metrics_uses_held_out_interactions(spark):
    recommendations = spark.createDataFrame(
        [(1, 10, 1), (1, 11, 2), (2, 10, 1)],
        ["user_id", "movie_id", "recommendation_rank"],
    )
    held_out = spark.createDataFrame(
        [(1, 10, 4.0), (1, 99, 3.0), (2, 20, 5.0)],
        ["user_id", "movie_id", "rating"],
    )

    metrics = calculate_ranking_metrics(recommendations, held_out, k=2)

    assert metrics["users_evaluated"] == 2
    assert metrics["relevant_recommendations"] == 1
    assert metrics["precision_at_k"] == 1 / 3
    assert metrics["recall_at_k"] == 0.25


def test_train_als_writes_metrics_and_loadable_artifact(spark, tmp_path):
    config = _config(tmp_path)
    _write_ratings(spark, tmp_path)

    metrics = train_als(spark, config)
    metrics_path = tmp_path / "models" / "model_metrics.json"
    model_path = tmp_path / "models" / "als"

    assert metrics["row_counts"]["training"] > 0
    assert metrics["row_counts"]["test"] > 0
    assert metrics["metrics"]["rmse"] >= 0
    assert metrics["metrics"]["mae"] >= 0
    assert metrics_path.is_file()
    assert model_path.is_dir()
    with metrics_path.open(encoding="utf-8") as metrics_file:
        saved_metrics = json.load(metrics_file)
    assert saved_metrics["parameters"]["algorithm"] == "spark_ml_als"
    assert load_als_model(config).userFactors.count() > 0