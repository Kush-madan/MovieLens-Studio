"""Tests for MovieLens Bronze ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recsys.ingestion import DataFileNotFoundError, EmptyInputError, ingest_all


def _config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "raw_ml1m": str(tmp_path / "raw"),
            "bronze": str(tmp_path / "bronze"),
            "silver": str(tmp_path / "silver"),
        }
    }


def _write_sample_inputs(raw_path: Path) -> None:
    raw_path.mkdir(parents=True, exist_ok=True)
    files = {
        "ratings.dat": "1::10::4.5::1700000000\n",
        "movies.dat": "10::Caf\xe9 (2001)::Comedy|Drama\n",
        "users.dat": "1::M::25::12::12345\n",
    }
    for name, contents in files.items():
        (raw_path / name).write_bytes(contents.encode("latin-1"))


def test_missing_file_raises_custom_error(spark, tmp_path):
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    (raw_path / "ratings.dat").write_text("1::10::4.5::1700000000\n", encoding="latin-1")

    with pytest.raises(DataFileNotFoundError, match="movies.dat"):
        ingest_all(spark, _config(tmp_path))


def test_empty_file_raises_custom_error(spark, tmp_path):
    raw_path = tmp_path / "raw"
    raw_path.mkdir()
    (raw_path / "ratings.dat").write_bytes(b"")

    with pytest.raises(EmptyInputError, match="ratings.dat"):
        ingest_all(spark, _config(tmp_path))


def test_valid_inputs_write_parquet_and_stats(spark, tmp_path):
    _write_sample_inputs(tmp_path / "raw")

    stats = ingest_all(spark, _config(tmp_path))
    bronze_path = tmp_path / "bronze"

    assert stats["tables"]["ratings"]["row_count"] == 1
    assert (bronze_path / "ratings").is_dir()
    assert (bronze_path / "movies").is_dir()
    assert (bronze_path / "users").is_dir()
    with (bronze_path / "ingestion_stats.json").open(encoding="utf-8") as stats_file:
        saved_stats = json.load(stats_file)
    assert saved_stats["tables"]["movies"]["file_size_bytes"] > 0
    assert spark.read.parquet(str(bronze_path / "movies")).first()["title"] == "Café (2001)"