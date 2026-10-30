# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every report names the quality config that produced the Tier 1 quality score."""

import json

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.reporting import BenchmarkReporter, JSONReporter, MarkdownReporter

_DIGEST = "sha256:" + "a" * 64


def _quality_result(quality_config: dict | None) -> ValidationResult:
    result = ValidationResult(validator_name="QUALITY", validator_description="Quality score")
    if quality_config is not None:
        result.metadata["quality_config"] = quality_config
    return result


def _benchmark_line(results: list[ValidationResult], field: str = "Quality config") -> str | None:
    card = BenchmarkReporter().render_all(results)
    return next((line for line in card.splitlines() if line.startswith(f"- {field}:")), None)


def test_json_report_records_quality_config() -> None:
    config = {"source": "custom", "file": "team-quality.yaml", "digest": _DIGEST, "min_score": 80}

    data = json.loads(JSONReporter().render_all([_quality_result(config)]))

    assert data["quality_config"] == config


def test_markdown_report_records_quality_config() -> None:
    report = MarkdownReporter().render_all(
        [_quality_result({"source": "custom", "file": "team-quality.yaml", "digest": _DIGEST, "min_score": 80})]
    )

    assert f"**Quality config:** custom (`team-quality.yaml`), `{_DIGEST}`, min score 80" in report


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param(
            {"source": "custom", "file": "team-quality.yaml", "digest": _DIGEST},
            f"- Quality config: custom (team-quality.yaml), `{_DIGEST}`",
            id="custom",
        ),
        pytest.param(
            {"source": "packaged", "file": None, "digest": _DIGEST},
            f"- Quality config: packaged, `{_DIGEST}`",
            id="packaged",
        ),
        pytest.param(
            {"source": "custom", "file": "team_*rules*.yaml", "digest": _DIGEST},
            f"- Quality config: custom (team\\_\\*rules\\*.yaml), `{_DIGEST}`",
            id="markdown-escaped-file-name",
        ),
    ],
)
def test_benchmark_records_quality_config(config: dict, expected: str) -> None:
    assert _benchmark_line([_quality_result(config)]) == expected


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(None, id="not-recorded"),
        pytest.param({"source": "custom", "file": "x.yaml", "digest": "sha256:`injected`"}, id="malformed-digest"),
        pytest.param({"source": "other", "file": "x.yaml", "digest": _DIGEST}, id="unknown-source"),
    ],
)
def test_benchmark_omits_missing_or_malformed_quality_config(config: dict | None) -> None:
    assert _benchmark_line([_quality_result(config)]) is None


def test_benchmark_records_quality_min_score() -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "min_score": 80}

    assert _benchmark_line([_quality_result(config)], "Quality min score") == "- Quality min score: 80"


@pytest.mark.parametrize("min_score", [None, True, "80", float("nan")], ids=["missing", "bool", "string", "nan"])
def test_benchmark_omits_missing_or_malformed_quality_min_score(min_score: object) -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "min_score": min_score}

    assert _benchmark_line([_quality_result(config)], "Quality min score") is None


_WEIGHTS = {"correctness": 0.5, "discoverability": 0.2, "reliability": 0.2, "efficiency": 0.1}
_WEIGHTS_LINE = "- Quality dimension weights: correctness 50%, discoverability 20%, reliability 20%, efficiency 10%"


def test_json_report_records_dimension_weights_and_minimums() -> None:
    config = {
        "source": "custom",
        "file": "team-quality.yaml",
        "digest": _DIGEST,
        "min_score": 70,
        "dimension_weights": _WEIGHTS,
        "dimension_minimums": {"reliability": 60},
    }

    data = json.loads(JSONReporter().render_all([_quality_result(config)]))

    assert data["quality_config"]["dimension_weights"] == _WEIGHTS
    assert data["quality_config"]["dimension_minimums"] == {"reliability": 60}


def test_markdown_report_records_dimension_weights_and_minimums() -> None:
    config = {
        "source": "custom",
        "file": "team-quality.yaml",
        "digest": _DIGEST,
        "min_score": 70,
        "dimension_weights": _WEIGHTS,
        "dimension_minimums": {"reliability": 60},
    }

    report = MarkdownReporter().render_all([_quality_result(config)])

    assert "**Quality dimension weights:** " in report
    assert "correctness 50%, discoverability 20%, reliability 20%, efficiency 10%" in report
    assert "**Quality dimension minimums:** reliability >= 60" in report


def test_markdown_report_omits_dimension_lines_for_legacy_config() -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "min_score": 70}

    report = MarkdownReporter().render_all([_quality_result(config)])

    assert "Quality dimension weights" not in report
    assert "Quality dimension minimums" not in report


def test_benchmark_records_dimension_weights_and_minimums() -> None:
    config = {
        "source": "custom",
        "file": "team-quality.yaml",
        "digest": _DIGEST,
        "dimension_weights": _WEIGHTS,
        "dimension_minimums": {"reliability": 60, "efficiency": 50.5},
    }
    card = BenchmarkReporter().render_all([_quality_result(config)])

    assert _WEIGHTS_LINE in card
    assert "- Quality dimension minimums: reliability >= 60, efficiency >= 50.5" in card


def test_benchmark_omits_dimension_minimums_when_none_configured() -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "dimension_weights": _WEIGHTS}

    card = BenchmarkReporter().render_all([_quality_result(config)])

    assert _WEIGHTS_LINE in card
    assert "Quality dimension minimums" not in card


@pytest.mark.parametrize(
    "weights",
    [
        None,
        {},
        {"correctness": 0.5},  # incomplete
        {"correctness": "x", "discoverability": 0.2, "reliability": 0.2, "efficiency": 0.1},
        {"correctness": -0.1, "discoverability": 0.3, "reliability": 0.4, "efficiency": 0.4},
    ],
    ids=["missing", "empty", "incomplete", "string", "negative"],
)
def test_benchmark_omits_missing_or_malformed_dimension_weights(weights: object) -> None:
    config = {"source": "packaged", "file": None, "digest": _DIGEST, "dimension_weights": weights}

    assert _benchmark_line([_quality_result(config)], "Quality dimension weights") is None


@pytest.mark.parametrize(
    "minimums",
    [
        {"reliability": 101},
        {"reliability": -1},
        {"reliability": "60"},
        {"quality": 60},
    ],
    ids=["above-100", "negative", "string", "unknown-dimension"],
)
def test_benchmark_omits_malformed_dimension_minimums(minimums: dict) -> None:
    config = {
        "source": "packaged",
        "file": None,
        "digest": _DIGEST,
        "dimension_weights": _WEIGHTS,
        "dimension_minimums": minimums,
    }

    assert _benchmark_line([_quality_result(config)], "Quality dimension minimums") is None
