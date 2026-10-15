"""Train and evaluate a PySpark MLlib ALS recommendation model."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .config import configured_path


class RatingsDatasetError(ValueError):
    """Raised when the Silver ratings dataset is missing or unusable."""


def _read_ratings(spark: SparkSession, config: dict[str, Any]) -> DataFrame:
    ratings_path = configured_path(config, "silver") / "ratings"
    if not ratings_path.is_dir():
        raise FileNotFoundError(f"Silver ratings dataset does not exist: {ratings_path}")
    ratings = spark.read.parquet(str(ratings_path)).select(
        F.col("user_id").cast("int").alias("user_id"),
        F.col("movie_id").cast("int").alias("movie_id"),
        F.col("rating").cast("float").alias("rating"),
    ).dropna(subset=["user_id", "movie_id", "rating"])
    if ratings.limit(1).count() == 0:
        raise RatingsDatasetError(f"Silver ratings dataset is empty: {ratings_path}")
    return ratings.filter((F.col("user_id") > 0) & (F.col("movie_id") > 0))


def _valid_prediction_frame(predictions: DataFrame) -> DataFrame:
    return predictions.filter(
        F.col("prediction").isNotNull() & ~F.isnan(F.col("prediction"))
    )


def evaluate_predictions(predictions: DataFrame) -> dict[str, float | int]:
    """Calculate RMSE and MAE after dropping null/NaN cold-start predictions."""
    valid_predictions = _valid_prediction_frame(predictions)
    prediction_count = valid_predictions.count()
    if prediction_count == 0:
        raise RatingsDatasetError("No valid ALS predictions were available for evaluation")
    rmse = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    ).evaluate(valid_predictions)
    mae = RegressionEvaluator(
        metricName="mae", labelCol="rating", predictionCol="prediction"
    ).evaluate(valid_predictions)
    return {
        "prediction_row_count": prediction_count,
        "rmse": float(rmse),
        "mae": float(mae),
    }


def calculate_ranking_metrics(
    recommendations: DataFrame, held_out: DataFrame, k: int
) -> dict[str, float | int]:
    """Calculate micro Precision@K and macro Recall@K from held-out interactions."""
    if k <= 0:
        raise ValueError("Ranking evaluation k must be positive")
    top_recommendations = recommendations.filter(F.col("recommendation_rank") <= k)
    held_out_pairs = held_out.select("user_id", "movie_id").dropDuplicates()
    recommendation_count = top_recommendations.count()
    relevant_count = top_recommendations.join(
        held_out_pairs, on=["user_id", "movie_id"], how="inner"
    ).count()
    relevant_by_user = held_out_pairs.groupBy("user_id").agg(
        F.count("movie_id").alias("held_out_count")
    )
    hits_by_user = top_recommendations.join(
        held_out_pairs, on=["user_id", "movie_id"], how="inner"
    ).groupBy("user_id").agg(F.count("movie_id").alias("hit_count"))
    recall = relevant_by_user.join(hits_by_user, "user_id", "left").fillna(
        0, subset=["hit_count"]
    ).select(F.avg(F.col("hit_count") / F.col("held_out_count")).alias("recall")).first()[0]
    return {
        "k": k,
        "users_evaluated": relevant_by_user.count(),
        "recommendations_evaluated": recommendation_count,
        "relevant_recommendations": relevant_count,
        "precision_at_k": (relevant_count / recommendation_count)
        if recommendation_count
        else 0.0,
        "recall_at_k": float(recall or 0.0),
    }


def evaluate_ranking(
    model: ALSModel, training: DataFrame, test: DataFrame, k: int
) -> dict[str, float | int]:
    """Evaluate ranking against held-out interactions without retraining ALS."""
    users = test.select("user_id").distinct()
    candidate_count = max(k * 10, k + 50)
    candidates = model.recommendForUserSubset(users, candidate_count).select(
        "user_id", F.explode("recommendations").alias("recommendation")
    ).select(
        "user_id",
        F.col("recommendation.movie_id").cast("int").alias("movie_id"),
        F.col("recommendation.rating").cast("double").alias("score"),
    ).join(
        training.select("user_id", "movie_id").dropDuplicates(),
        on=["user_id", "movie_id"],
        how="left_anti",
    )
    recommendations = candidates.withColumn(
        "recommendation_rank",
        F.row_number().over(
            Window.partitionBy("user_id").orderBy(F.col("score").desc(), "movie_id")
        ),
    )
    return calculate_ranking_metrics(recommendations, test, k)


def train_als(spark: SparkSession, config: dict[str, Any]) -> dict[str, Any]:
    """Train ALS on Silver ratings, save its artifact, and write actual metrics."""
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("The model configuration must be a YAML mapping")
    rank = int(model_config.get("rank", 10))
    max_iter = int(model_config.get("max_iter", 10))
    reg_param = float(model_config.get("reg_param", 0.1))
    seed = int(model_config.get("seed", config.get("project", {}).get("random_seed", 42)))
    train_fraction = float(model_config.get("train_fraction", 0.8))
    ranking_k = int(model_config.get("ranking_k", 10))
    if rank <= 0 or max_iter <= 0 or reg_param < 0:
        raise ValueError("ALS rank and max_iter must be positive; reg_param cannot be negative")
    if not 0 < train_fraction < 1 or ranking_k <= 0:
        raise ValueError("model.train_fraction must be between zero and one")

    ratings = _read_ratings(spark, config).cache()
    total_rows = ratings.count()
    training, test = ratings.randomSplit(
        [train_fraction, 1.0 - train_fraction], seed=seed
    )
    training_count = training.count()
    test_count = test.count()
    if training_count == 0 or test_count == 0:
        raise RatingsDatasetError(
            f"Train/test split produced empty data: training={training_count}, test={test_count}"
        )

    als = ALS(
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        seed=seed,
        coldStartStrategy="drop",
        nonnegative=False,
    )
    model = als.fit(training)
    metrics = evaluate_predictions(model.transform(test))
    ranking_metrics = evaluate_ranking(model, training, test, ranking_k)

    model_path = configured_path(config, "models") / "als"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_path))
    metrics_path = configured_path(config, "models") / "model_metrics.json"
    metrics_record: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "parameters": {
            "algorithm": "spark_ml_als",
            "rank": rank,
            "max_iter": max_iter,
            "reg_param": reg_param,
            "seed": seed,
            "train_fraction": train_fraction,
            "ranking_k": ranking_k,
            "cold_start_strategy": "drop",
        },
        "row_counts": {
            "ratings_total": total_rows,
            "training": training_count,
            "test": test_count,
            "predictions_evaluated": metrics["prediction_row_count"],
        },
        "metrics": {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "ranking": ranking_metrics,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_record, metrics_file, indent=2, sort_keys=True)
    ratings.unpersist()
    return metrics_record


def load_als_model(config: dict[str, Any]) -> ALSModel:
    """Load the saved ALS model or raise a clear artifact error."""
    model_path = configured_path(config, "models") / "als"
    if not model_path.is_dir():
        raise FileNotFoundError(f"ALS model artifact does not exist: {model_path}")
    return ALSModel.load(str(model_path))