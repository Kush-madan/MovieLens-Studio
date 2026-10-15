"""Low-memory Spark session creation for local Windows development."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession

from .config import PROJECT_ROOT, load_config, resolve_project_path


def create_spark_session(
    configuration: dict[str, Any] | None = None,
    app_name: str | None = None,
) -> SparkSession:
    """Create a local Spark session using the project's resource limits."""
    configuration = configuration or load_config()
    spark_settings = configuration.get("spark", {})
    if not isinstance(spark_settings, dict):
        raise ValueError("The spark configuration must be a YAML mapping")

    windows_settings = spark_settings.get("windows", {})
    if not isinstance(windows_settings, dict):
        raise ValueError("The spark.windows configuration must be a YAML mapping")

    configured_hadoop_home = windows_settings.get("hadoop_home")
    if configured_hadoop_home:
        hadoop_home = resolve_project_path(configured_hadoop_home)
        if not hadoop_home.is_dir():
            raise FileNotFoundError(
                f"Configured HADOOP_HOME does not exist: {hadoop_home}. "
                "Set it to a directory containing winutils.exe, or remove it "
                "when your Spark installation does not need it."
            )
        os.environ["HADOOP_HOME"] = str(hadoop_home)

    checkpoint_dir = resolve_project_path(
        spark_settings.get("checkpoint_dir", "data/.spark_checkpoint")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(
            app_name or spark_settings.get("app_name", "MovieRecommendationSystem")
        )
        .master(spark_settings.get("master", "local[2]"))
        .config("spark.driver.memory", spark_settings.get("driver_memory", "1g"))
        .config("spark.executor.memory", spark_settings.get("executor_memory", "1g"))
        .config(
            "spark.sql.shuffle.partitions",
            str(spark_settings.get("shuffle_partitions", 4)),
        )
        .config(
            "spark.sql.adaptive.enabled",
            str(spark_settings.get("adaptive_query_execution", True)).lower(),
        )
        .config("spark.sql.warehouse.dir", str(PROJECT_ROOT / "data" / "spark-warehouse"))
        .config("spark.local.dir", str(PROJECT_ROOT / "data" / ".spark_local"))
        .config("spark.checkpoint.dir", str(checkpoint_dir))
    )

    session = builder.getOrCreate()
    session.sparkContext.setLogLevel(spark_settings.get("log_level", "WARN"))
    return session
