"""Airflow orchestration for the existing MovieLens Studio pipeline."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recsys.config import load_config  # noqa: E402
from recsys.pipeline import run_stage  # noqa: E402
from recsys.pipeline_validation import validate_pipeline_outputs  # noqa: E402
from recsys.spark_session import create_spark_session  # noqa: E402


def _run_existing_stage(stage: str) -> None:
    """Call an existing stage wrapper; no stage logic is duplicated here."""
    run_stage(stage, load_config())


def _validate_outputs() -> dict:
    config = load_config()
    spark = create_spark_session(config, "MovieRecommendationSystem-Validation")
    try:
        return validate_pipeline_outputs(spark, config)
    finally:
        spark.stop()


default_args = {
    "owner": "movie-recsys",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="movielens_recommendation_pipeline",
    description="Orchestrate the existing MovieLens Bronze-to-recommendations pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    dagrun_timeout=timedelta(hours=4),
    max_active_runs=1,
    tags=["movielens", "pyspark", "recommendations"],
) as dag:
    ingest_bronze = PythonOperator(
        task_id="ingest_bronze",
        python_callable=_run_existing_stage,
        op_kwargs={"stage": "ingest_bronze"},
        execution_timeout=timedelta(hours=1),
        retries=1,
    )
    clean_silver = PythonOperator(
        task_id="clean_silver",
        python_callable=_run_existing_stage,
        op_kwargs={"stage": "clean_silver"},
        execution_timeout=timedelta(hours=1),
        retries=1,
    )
    build_gold_analytics = PythonOperator(
        task_id="build_gold_analytics",
        python_callable=_run_existing_stage,
        op_kwargs={"stage": "build_gold_analytics"},
        execution_timeout=timedelta(hours=1),
        retries=1,
    )
    train_als_model = PythonOperator(
        task_id="train_als_model",
        python_callable=_run_existing_stage,
        op_kwargs={"stage": "train_als_model"},
        execution_timeout=timedelta(hours=2),
        retries=1,
    )
    generate_recommendations = PythonOperator(
        task_id="generate_recommendations",
        python_callable=_run_existing_stage,
        op_kwargs={"stage": "generate_recommendations"},
        execution_timeout=timedelta(hours=2),
        retries=1,
    )
    validate_pipeline_outputs_task = PythonOperator(
        task_id="validate_pipeline_outputs",
        python_callable=_validate_outputs,
        execution_timeout=timedelta(minutes=30),
        retries=0,
    )

    ingest_bronze >> clean_silver >> build_gold_analytics >> train_als_model
    train_als_model >> generate_recommendations >> validate_pipeline_outputs_task