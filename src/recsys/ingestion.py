"""Ingest MovieLens 1M text files into the Bronze Parquet layer."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from .config import configured_path


LOGGER = logging.getLogger(__name__)


class DataFileNotFoundError(FileNotFoundError):
    """Raised when a required MovieLens input file is missing."""


class EmptyInputError(ValueError):
    """Raised when a required MovieLens input file contains no records."""


TABLE_DEFINITIONS: dict[str, tuple[str, list[str]]] = {
    "ratings": (
        "ratings.dat",
        ["user_id", "movie_id", "rating", "timestamp"],
    ),
    "movies": (
        "movies.dat",
        ["movie_id", "title", "genres"],
    ),
    "users": (
        "users.dat",
        ["user_id", "gender", "age", "occupation", "zip_code"],
    ),
}


def _validate_input_file(path: Path) -> None:
    if not path.is_file():
        raise DataFileNotFoundError(
            f"Required MovieLens input file was not found: {path}"
        )
    if path.stat().st_size == 0:
        raise EmptyInputError(f"MovieLens input file is empty: {path}")


def _read_table(spark: SparkSession, path: Path, columns: list[str]) -> DataFrame:
    _validate_input_file(path)
    raw = (
        spark.read.format("csv")
        .schema(StructType([StructField("value", StringType(), True)]))
        .option("encoding", "ISO-8859-1")
        .option("delimiter", "\u0001")
        .option("quote", "")
        .option("escape", "")
        .load(str(path))
    )
    if raw.limit(1).count() == 0:
        raise EmptyInputError(f"MovieLens input file contains no records: {path}")

    values = F.split(F.col("value"), "::", -1)
    return raw.select(
        *[values.getItem(index).alias(column) for index, column in enumerate(columns)]
    )


def _write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, indent=2, sort_keys=True)


def ingest_all(spark: SparkSession, config: dict[str, Any]) -> dict[str, Any]:
    """Ingest all MovieLens files into Bronze and return ingestion statistics."""
    raw_path = configured_path(config, "raw_ml1m")
    bronze_path = configured_path(config, "bronze")
    bronze_path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()
    table_stats: dict[str, dict[str, Any]] = {}

    for table_name, (file_name, columns) in TABLE_DEFINITIONS.items():
        source_path = raw_path / file_name
        table_started = time.perf_counter()
        frame = _read_table(spark, source_path, columns)
        frame = frame.withColumn("ingested_at", F.current_timestamp()).withColumn(
            "source_file", F.lit(file_name)
        )
        row_count = frame.count()
        if row_count == 0:
            raise EmptyInputError(f"MovieLens input file contains no records: {source_path}")

        output_path = bronze_path / table_name
        frame.write.mode("overwrite").parquet(str(output_path))
        table_stats[table_name] = {
            "row_count": row_count,
            "file_size_bytes": source_path.stat().st_size,
            "duration_seconds": round(time.perf_counter() - table_started, 3),
            "source_file": file_name,
        }

    stats = {
        "ingested_at": ingestion_timestamp,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "tables": table_stats,
    }
    _write_stats(bronze_path / "ingestion_stats.json", stats)
    LOGGER.info("MovieLens Bronze ingestion completed: %s", stats)
    return stats