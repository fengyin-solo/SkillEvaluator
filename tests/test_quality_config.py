# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from skillevaluator.quality_config import (
    QualityConfigError,
    format_dimension_minimums,
    format_dimension_weights,
    load_quality_config,
    quality_config_digest,
    quality_config_provenance,
)


def test_packaged_default_quality_config_is_available() -> None:
    config = load_quality_config()

    assert config["expected_headers"] == [
        {
            "heading": "## Prerequisites",
            "alternatives": ["## Requirements"],
            "points": 5,
            "message": "No prerequisites/requirements documented",
            "suggestion": "Document dependencies, API keys, or setup needed",
        },
        {
            "heading": "## Limitations",
            "alternatives": [],
            "points": 5,
            "message": "No limitations documented",
            "suggestion": "Add '## Limitations' section with known issues/constraints",
        },
        {
            "heading": "## Troubleshooting",
            "alternatives": ["## Common Issues", "## FAQ"],
            "points": 5,
            "message": "No troubleshooting section documented",
            "suggestion": "Add '## Troubleshooting' with Error/Cause/Solution patterns",
        },
    ]
    # Upstream deducted the int 5, so reports keep "deduction": 5 rather than 5.0.
    assert all(type(rule["points"]) is int for rule in config["expected_headers"])


def test_quality_config_can_be_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'expected_headers:\n  - heading: "## Failure Constellations"\n'
        '    alternatives:\n      - "## Known Failures"\n'
        "    message: No failure constellations documented\n"
        "    suggestion: Add '## Failure Constellations'\n"
        '  - heading: "### Security"\n    points: 8.5\n'
        "    message: No security notes documented\n"
        "    suggestion: Add '### Security'\n",
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert config["expected_headers"] == [
        {
            "heading": "## Failure Constellations",
            "alternatives": ["## Known Failures"],
            "points": 5,
            "message": "No failure constellations documented",
            "suggestion": "Add '## Failure Constellations'",
        },
        {
            "heading": "### Security",
            "alternatives": [],
            "points": 8.5,
            "message": "No security notes documented",
            "suggestion": "Add '### Security'",
        },
    ]


@pytest.mark.parametrize(
    "config",
    [
        "expected_headers: not-a-list\n",
        "expected_headers:\n  - heading: Limitations\n",
        'expected_headers:\n  - heading: "##   "\n',
        'expected_headers:\n  - heading: "## Limitations"\n    alternatives: not-a-list\n',
        'expected_headers:\n  - heading: "## Limitations"\n    points: -1\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: ""\n',
        'expected_headers:\n  - heading: "## Limitations"\n    suggestion:\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: [Missing]\n',
    ],
)
def test_quality_config_rejects_invalid_expected_headers(tmp_path: Path, config: str) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match="expected_headers"):
        load_quality_config(config_path)


def test_quality_config_generates_message_and_suggestion_when_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'expected_headers:\n  - heading: "## Security"\n'
        '  - heading: "## Prerequisites"\n    alternatives:\n      - "## Requirements"\n    points: 5\n'
        '  - heading: "## Limitations"\n    message: No limitations documented\n',
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert [(rule["message"], rule["suggestion"]) for rule in config["expected_headers"]] == [
        ("Missing expected heading: '## Security'", "Add a '## Security' section"),
        (
            "Missing expected heading: '## Prerequisites' or '## Requirements'",
            "Add one of these sections: '## Prerequisites' or '## Requirements'",
        ),
        ("No limitations documented", "Add a '## Limitations' section"),
    ]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            "expected_headers: []\nexpected_header: []\n",
            r"unknown key\(s\) in top level: expected_header",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    point: 10\n'
            "    message: Missing\n    suggestion: Add it\n",
            r"unknown key\(s\) in expected_headers\[0\]: point",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    message: Missing\n    suggestion: Add it\n'
            '  - heading: "## Limits"\n    alternatives:\n      - "##  Limitations"\n'
            "    message: Missing\n    suggestion: Add it\n",
            r"duplicate heading '## Limitations' in expected_headers\[1\] \(already listed in expected_headers\[0\]\)",
        ),
    ],
)
def test_quality_config_rejects_unknown_keys_and_duplicate_headings(tmp_path: Path, config: str, match: str) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(config_path)


def test_quality_config_errors_name_the_file_and_yaml_position(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: [\n", encoding="utf-8")

    with pytest.raises(QualityConfigError) as exc_info:
        load_quality_config(config_path)

    message = str(exc_info.value)
    assert message.startswith(f"{config_path}: invalid YAML: line 2, column 1: ")
    assert "\n" not in message


def test_quality_config_rejects_non_utf8_file(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: []\n", encoding="utf-16")

    with pytest.raises(QualityConfigError, match="not valid UTF-8"):
        load_quality_config(config_path)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            'expected_headers:\n  - heading: "## Prerequisites"\nexpected_headers:\n  - heading: "## Security"\n',
            r"invalid YAML: line 3, column 1: duplicate key 'expected_headers'",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: 5\n    points: 10\n',
            r"invalid YAML: line 4, column 5: duplicate key 'points'",
        ),
        ("expected_headers: " + "[" * 10000 + "]" * 10000 + "\n", "invalid YAML: nesting is too deep"),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: ' + "9" * 400 + "\n",
            r"expected_headers\[0\]\.points must be a non-negative number",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: ' + "9" * 5000 + "\n",
            r"invalid YAML: cannot parse a value: Exceeds the limit",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    message: 2026-02-30\n',
            r"invalid YAML: cannot parse a value: day is out of range",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    points: !!int abc\n',
            r"invalid YAML: cannot parse a value: invalid literal for int\(\)",
        ),
    ],
    ids=[
        "duplicate-top-level-key",
        "duplicate-rule-key",
        "deep-nesting",
        "points-overflow",
        "integer-digit-limit",
        "impossible-date",
        "bad-explicit-tag",
    ],
)
def test_quality_config_rejects_duplicate_keys_and_pathological_values(tmp_path: Path, config: str, match: str) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(config_path)


def test_quality_config_digest_ignores_formatting_but_not_rules(tmp_path: Path) -> None:
    compact = tmp_path / "compact.yaml"
    compact.write_text('expected_headers: [{heading: "## Limitations", points: 5}]\n', encoding="utf-8")
    verbose = tmp_path / "verbose.yaml"
    verbose.write_text(
        '# team rules\nexpected_headers:\n  - heading:   "##   Limitations"\n    points: 5\n',
        encoding="utf-8",
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text('expected_headers:\n  - heading: "## Limitations"\n    points: 6\n', encoding="utf-8")

    digest = quality_config_digest(load_quality_config(compact))

    assert digest.startswith("sha256:") and len(digest) == 71
    assert quality_config_digest(load_quality_config(verbose)) == digest
    assert quality_config_digest(load_quality_config(changed)) != digest


def test_quality_config_provenance_names_custom_file_without_its_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "team-quality.yaml"
    config_path.write_text("expected_headers: []\n", encoding="utf-8")
    config = load_quality_config(config_path)

    assert quality_config_provenance(config_path, config) == {
        "source": "custom",
        "file": "team-quality.yaml",
        "digest": quality_config_digest(config),
    }
    assert quality_config_provenance(None, load_quality_config())["file"] is None


_WEIGHTS_HEADER = "expected_headers: []\ndimension_weights:\n"
_MINIMUMS_HEADER = "expected_headers: []\ndimension_minimums:\n"


def _write_config(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_packaged_defaults_fill_dimension_weights_and_minimums() -> None:
    config = load_quality_config()

    assert config["dimension_weights"] == {
        "correctness": 0.35,
        "discoverability": 0.25,
        "reliability": 0.25,
        "efficiency": 0.15,
    }
    assert config["dimension_minimums"] == {}


def test_custom_config_without_dimension_blocks_keeps_default_weights(tmp_path: Path) -> None:
    config = load_quality_config(_write_config(tmp_path, "expected_headers: []\n"))

    assert config["dimension_weights"] == pytest.approx(
        {"correctness": 0.35, "discoverability": 0.25, "reliability": 0.25, "efficiency": 0.15}
    )
    assert config["dimension_minimums"] == {}


def test_dimension_weights_are_normalized_in_canonical_order(tmp_path: Path) -> None:
    body = (
        "expected_headers: []\n"
        "dimension_weights:\n"
        "  efficiency: 0.15\n"
        "  reliability: 0.25\n"
        "  discoverability: 0.25\n"
        "  correctness: 0.35\n"
    )
    config = load_quality_config(_write_config(tmp_path, body))

    assert list(config["dimension_weights"]) == [
        "correctness",
        "discoverability",
        "reliability",
        "efficiency",
    ]
    assert config["dimension_weights"] == pytest.approx(
        {"correctness": 0.35, "discoverability": 0.25, "reliability": 0.25, "efficiency": 0.15}
    )


def test_dimension_minimums_accept_any_subset(tmp_path: Path) -> None:
    body = _MINIMUMS_HEADER + "  reliability: 60\n  efficiency: 50.5\n"
    config = load_quality_config(_write_config(tmp_path, body))

    assert config["dimension_minimums"] == {"reliability": 60.0, "efficiency": 50.5}


def test_empty_dimension_minimums_mapping_is_allowed(tmp_path: Path) -> None:
    config = load_quality_config(_write_config(tmp_path, "expected_headers: []\ndimension_minimums: {}\n"))

    assert config["dimension_minimums"] == {}


def test_decimal_weights_with_float_buildup_still_sum_to_one(tmp_path: Path) -> None:
    # 0.1 + 0.2 + 0.3 + 0.4 is 0.9999999999999999 as floats; the tolerance must
    # accept it rather than rejecting an obviously valid configuration.
    body = _WEIGHTS_HEADER + "  correctness: 0.1\n  discoverability: 0.2\n  reliability: 0.3\n  efficiency: 0.4\n"
    config_path = _write_config(tmp_path, body)

    config = load_quality_config(config_path)
    assert sum(config["dimension_weights"].values()) == pytest.approx(1.0)
    # Reparsing the same file yields bit-identical weights, so conclusions cannot drift.
    assert load_quality_config(config_path)["dimension_weights"] == config["dimension_weights"]


def test_zero_weights_are_allowed(tmp_path: Path) -> None:
    body = _WEIGHTS_HEADER + "  correctness: 1.0\n  discoverability: 0\n  reliability: 0.0\n  efficiency: 0\n"
    config = load_quality_config(_write_config(tmp_path, body))

    assert config["dimension_weights"]["discoverability"] == 0.0
    assert config["dimension_weights"]["efficiency"] == 0.0


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: 0.35\n  reliability: 0.25\n  efficiency: 0.15\n",
            r"dimension_weights must set all four dimensions; missing: discoverability",
            id="one-missing",
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: 0.5\n  efficiency: 0.5\n",
            r"dimension_weights must set all four dimensions; missing: discoverability, reliability",
            id="two-missing",
        ),
        pytest.param("expected_headers: []\ndimension_weights: {}\n", r"missing: ", id="empty-mapping"),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: 0.35\n  discoverability: 0.25\n"
            "  reliability: 0.25\n  efficiency: 0.15\n  quality: 0.0\n",
            r"unknown key\(s\) in dimension_weights: quality",
            id="unknown-dimension",
        ),
        pytest.param(
            "expected_headers: []\ndimension_weights: [0.35]\n", r"dimension_weights must be a mapping", id="list"
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: -0.1\n  discoverability: 0.35\n  reliability: 0.35\n  efficiency: 0.4\n",
            r"dimension_weights\.correctness must be a finite non-negative number",
            id="negative",
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: true\n  discoverability: 0.2\n  reliability: 0.4\n  efficiency: 0.4\n",
            r"dimension_weights\.correctness must be a finite non-negative number",
            id="boolean",
        ),
        pytest.param(
            _WEIGHTS_HEADER + '  correctness: "0.35"\n  discoverability: 0.25\n'
            "  reliability: 0.25\n  efficiency: 0.15\n",
            r"dimension_weights\.correctness must be a finite non-negative number",
            id="string",
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: .nan\n  discoverability: 0.25\n  reliability: 0.25\n  efficiency: 0.15\n",
            r"dimension_weights\.correctness must be a finite non-negative number",
            id="nan",
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: .inf\n  discoverability: 0.25\n  reliability: 0.25\n  efficiency: 0.15\n",
            r"dimension_weights\.correctness must be a finite non-negative number",
            id="infinity",
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: 0.3\n  discoverability: 0.2\n  reliability: 0.3\n  efficiency: 0.1\n",
            r"dimension_weights must sum to 1\.0 \(got 0\.9\)",
            id="wrong-sum",
        ),
        pytest.param(
            _WEIGHTS_HEADER + "  correctness: 0.5\n  discoverability: 0.5\n  reliability: 0.5\n  efficiency: 0.5\n",
            r"dimension_weights must sum to 1\.0 \(got 2\)",
            id="sum-two",
        ),
    ],
)
def test_dimension_weights_reject_invalid_values(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(_write_config(tmp_path, body))


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            _MINIMUMS_HEADER + "  reliability: 101\n",
            r"dimension_minimums\.reliability must be a number between 0 and 100",
            id="above-100",
        ),
        pytest.param(
            _MINIMUMS_HEADER + "  reliability: -1\n",
            r"dimension_minimums\.reliability must be a number between 0 and 100",
            id="negative",
        ),
        pytest.param(
            _MINIMUMS_HEADER + '  reliability: "60"\n',
            r"dimension_minimums\.reliability must be a number between 0 and 100",
            id="string",
        ),
        pytest.param(
            _MINIMUMS_HEADER + "  reliability: true\n",
            r"dimension_minimums\.reliability must be a number between 0 and 100",
            id="boolean",
        ),
        pytest.param(
            _MINIMUMS_HEADER + "  reliability: .nan\n",
            r"dimension_minimums\.reliability must be a number between 0 and 100",
            id="nan",
        ),
        pytest.param(
            "expected_headers: []\ndimension_minimums: [60]\n",
            r"dimension_minimums must be a mapping",
            id="list",
        ),
        pytest.param(
            _MINIMUMS_HEADER + "  quality: 60\n",
            r"unknown key\(s\) in dimension_minimums: quality",
            id="unknown-dimension",
        ),
    ],
)
def test_dimension_minimums_reject_invalid_values(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(_write_config(tmp_path, body))


def test_dimension_minimum_boundaries_zero_and_one_hundred_are_allowed(tmp_path: Path) -> None:
    body = _MINIMUMS_HEADER + "  correctness: 0\n  reliability: 100\n"
    config = load_quality_config(_write_config(tmp_path, body))

    assert config["dimension_minimums"] == {"correctness": 0.0, "reliability": 100.0}


def test_duplicate_dimension_key_is_a_yaml_error(tmp_path: Path) -> None:
    body = (
        _WEIGHTS_HEADER + "  correctness: 0.35\n  correctness: 0.4\n"
        "  discoverability: 0.2\n  reliability: 0.25\n  efficiency: 0.15\n"
    )
    with pytest.raises(QualityConfigError, match=r"invalid YAML: .*duplicate key 'correctness'"):
        load_quality_config(_write_config(tmp_path, body))


def test_unknown_top_level_dimension_block_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(QualityConfigError, match=r"unknown key\(s\) in top level: dimension_threshold"):
        load_quality_config(_write_config(tmp_path, "expected_headers: []\ndimension_threshold: {}\n"))


def test_normalized_config_is_immutable(tmp_path: Path) -> None:
    config = load_quality_config(_write_config(tmp_path, _MINIMUMS_HEADER + "  reliability: 60\n"))

    with pytest.raises(TypeError):
        config["dimension_weights"]["correctness"] = 0.5
    with pytest.raises(TypeError):
        config["dimension_minimums"]["reliability"] = 90
    with pytest.raises(TypeError):
        config["new_key"] = 1


def test_dimension_changes_change_the_digest_but_formatting_does_not(tmp_path: Path) -> None:
    plain = _write_config(
        tmp_path,
        _WEIGHTS_HEADER + "  correctness: 0.1\n  discoverability: 0.2\n  reliability: 0.3\n  efficiency: 0.4\n",
    )
    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(
        "expected_headers: []\ndimension_weights:\n"
        "  efficiency: 0.4\n  reliability: 0.3\n  discoverability: 0.2\n  correctness: 0.1\n",
        encoding="utf-8",
    )
    changed_weights = tmp_path / "weights.yaml"
    changed_weights.write_text(
        _WEIGHTS_HEADER + "  correctness: 0.2\n  discoverability: 0.2\n  reliability: 0.3\n  efficiency: 0.3\n",
        encoding="utf-8",
    )
    changed_minimums = tmp_path / "minimums.yaml"
    changed_minimums.write_text("expected_headers: []\ndimension_minimums:\n  reliability: 60\n", encoding="utf-8")
    plain_default = tmp_path / "plain-default.yaml"
    plain_default.write_text("expected_headers: []\n", encoding="utf-8")

    default_digest = quality_config_digest(load_quality_config(plain_default))
    weights_digest = quality_config_digest(load_quality_config(plain))

    assert quality_config_digest(load_quality_config(reordered)) == weights_digest
    assert quality_config_digest(load_quality_config(changed_weights)) != weights_digest
    assert quality_config_digest(load_quality_config(changed_minimums)) != default_digest


def test_dimension_formatters_render_in_canonical_order() -> None:
    assert (
        format_dimension_weights(
            {"efficiency": 0.15, "reliability": 0.25, "discoverability": 0.25, "correctness": 0.35}
        )
        == "correctness 35%, discoverability 25%, reliability 25%, efficiency 15%"
    )
    assert format_dimension_minimums({"reliability": 60.0, "efficiency": 50.5}) == (
        "reliability >= 60, efficiency >= 50.5"
    )
