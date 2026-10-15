"""Command-line orchestration for running the existing pipeline functions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .analytics import analytics_all
from .cleaning import clean_all
from .config import load_config
from .ingestion import ingest_all
from .model import train_als
from .recommend import generate_recommendations
from .spark_session import create_spark_session


STAGES: dict[str, Callable[..., Any]] = {
    "ingest_bronze": ingest_all,
    "clean_silver": clean_all,
    "build_gold_analytics": analytics_all,
    "train_als_model": train_als,
}


def run_stage(stage: str, config: dict[str, Any]) -> Any:
    """Run one existing stage function with a short-lived Spark session."""
    if stage == "generate_recommendations":
        function = generate_recommendations
    elif stage in STAGES:
        function = STAGES[stage]
    else:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    spark = create_spark_session(config, f"MovieRecommendationSystem-{stage}")
    try:
        return function(spark, config)
    finally:
        spark.stop()


def run_pipeline(config: dict[str, Any]) -> None:
    """Run the existing pipeline functions in dependency order."""
    for stage in (
        "ingest_bronze",
        "clean_silver",
        "build_gold_analytics",
        "train_als_model",
        "generate_recommendations",
    ):
        run_stage(stage, config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MovieLens Studio pipeline stages")
    parser.add_argument(
        "stage",
        choices=[*STAGES, "generate_recommendations", "all"],
        help="Stage to run, or all stages in dependency order",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "all":
        run_pipeline(config)
    else:
        run_stage(args.stage, config)


if __name__ == "__main__":
    main()