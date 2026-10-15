"""Static and optional runtime checks for the Airflow DAG."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


DAG_PATH = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "movie_recsys_dag.py"


def test_dag_declares_required_tasks_and_dependencies():
    source = DAG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    task_ids = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in {
            "ingest_bronze",
            "clean_silver",
            "build_gold_analytics",
            "train_als_model",
            "generate_recommendations",
            "validate_pipeline_outputs",
        }
    }
    assert len(task_ids) == 6
    assert "ingest_bronze >> clean_silver >> build_gold_analytics >> train_als_model" in source
    assert "train_als_model >> generate_recommendations >> validate_pipeline_outputs_task" in source
    assert "schedule=None" in source


def test_dag_imports_and_dependencies_when_airflow_is_available():
    try:
        from airflow import DAG as _AirflowDAG  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("Apache Airflow is not installed in this environment")
    import importlib.util

    spec = importlib.util.spec_from_file_location("movie_recsys_dag", DAG_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.dag.task_ids == [
        "ingest_bronze",
        "clean_silver",
        "build_gold_analytics",
        "train_als_model",
        "generate_recommendations",
        "validate_pipeline_outputs",
    ]