"""Tests for MovieLens Silver cleaning."""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, TimestampType

from recsys.cleaning import clean_all
from recsys.ingestion import ingest_all


def _config(tmp_path: Path) -> dict:
    return {
        "cleaning": {"rating_min": 0.5, "rating_max": 5.0},
        "paths": {
            "raw_ml1m": str(tmp_path / "raw"),
            "bronze": str(tmp_path / "bronze"),
            "silver": str(tmp_path / "silver"),
        },
    }


def _write_inputs(raw_path: Path) -> None:
    raw_path.mkdir(parents=True, exist_ok=True)
    ratings = "\n".join(
        [
            "1::10::2.0::1700000000",
            "1::10::4.5::1700000100",
            "2::20::3.0::1700000200",
            "2::20::3.0::1700000200",
            "3::10::6.0::1700000300",
            "::10::4.0::1700000400",
            "4::999::4.0::1700000500",
        ]
    ) + "\n"
    (raw_path / "ratings.dat").write_text(ratings, encoding="latin-1")
    (raw_path / "movies.dat").write_text(
        "10::Movie One (2000)::Comedy|Drama\n20::Movie Two (2001)::(no genres listed)\n",
        encoding="latin-1",
    )
    (raw_path / "users.dat").write_text(
        "1::M::25::12::12345\n2::F::35::1::54321\n", encoding="latin-1"
    )


def _cleaned(spark, tmp_path: Path):
    config = _config(tmp_path)
    _write_inputs(tmp_path / "raw")
    ingest_all(spark, config)
    clean_all(spark, config)
    return spark.read.parquet(str(tmp_path / "silver" / "ratings")), config


def test_invalid_ratings_null_ids_and_orphans_are_removed(spark, tmp_path):
    ratings, config = _cleaned(spark, tmp_path)

    assert ratings.count() == 2
    assert {(row.user_id, row.movie_id) for row in ratings.collect()} == {(1, 10), (2, 20)}
    with (tmp_path / "silver" / "data_quality_stats.json").open(encoding="utf-8") as stats_file:
        stats = json.load(stats_file)
    assert stats["per_rule_rejected_counts"]["rating_out_of_range"] == 1
    assert stats["per_rule_rejected_counts"]["null_required_fields"] == 1
    assert stats["orphan_count"] == 1
    assert stats["rows_out"] == 2
    assert config["cleaning"]["rating_min"] == 0.5


def test_exact_duplicates_and_pair_duplicates_keep_latest_rating(spark, tmp_path):
    ratings, _ = _cleaned(spark, tmp_path)

    latest = ratings.filter("user_id = 1 AND movie_id = 10").first()
    assert latest.rating == 4.5
    with (tmp_path / "silver" / "data_quality_stats.json").open(encoding="utf-8") as stats_file:
        stats = json.load(stats_file)
    assert stats["tables"]["ratings"]["exact_duplicate_rows"] == 1
    assert stats["tables"]["ratings"]["duplicate_user_movie_rows"] == 1


def test_cleaned_output_has_expected_schema_and_movie_enrichment(spark, tmp_path):
    ratings, _ = _cleaned(spark, tmp_path)
    movies = spark.read.parquet(str(tmp_path / "silver" / "movies"))

    rating_types = {field.name: field.dataType for field in ratings.schema.fields}
    assert isinstance(rating_types["user_id"], IntegerType)
    assert isinstance(rating_types["movie_id"], IntegerType)
    assert isinstance(rating_types["rating"], DoubleType)
    assert isinstance(rating_types["timestamp"], TimestampType)

    movie_types = {field.name: field.dataType for field in movies.schema.fields}
    assert isinstance(movie_types["movie_id"], IntegerType)
    assert isinstance(movie_types["release_year"], IntegerType)
    assert isinstance(movie_types["genres"], ArrayType)
    movie_two = movies.filter("movie_id = 20").first()
    assert movie_two.release_year == 2001
    assert movie_two.genres == []