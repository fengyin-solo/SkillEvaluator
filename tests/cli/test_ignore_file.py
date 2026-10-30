# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for --ignore-file and skillevaluator-ignore.yaml discovery."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from skillevaluator.cli import _catalog_child_argv_from_ctx, cli
from skillevaluator.utils.rich_markup import strip_terminal_controls

_SKILL_MD = """---
name: frontmatter-name
description: A skill used to verify the ignore-file gate behavior under validation tests.
metadata:
  author: Gate Test <gate@example.com>
---

# Gate Skill

A body paragraph with enough words to be presentable.

## Instructions

Do the thing and verify the outcome before continuing forward.
"""


@pytest.fixture
def mismatched_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "gate-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    return skill_dir


_IGNORE_DOCUMENT = """
version: 1
ignores:
  - id: rename-batch
    check: SCHEMA.name_consistency
    paths: [SKILL.md]
    reason: accepted for the tracked rename batch
    expires: 2030-01-01
"""


def _json_report(output_dir: Path) -> dict:
    """Read the plain JSON report (``*.sarif.json`` is a different format)."""
    candidates = [path for path in output_dir.glob("*.json") if not path.name.endswith(".sarif.json")]
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def _invoke(skill_dir: Path, output_dir: Path, *extra: str) -> object:
    return CliRunner().invoke(
        cli,
        [
            "validate",
            str(skill_dir),
            "--checks",
            "schema",
            "--no-tier3",
            "--no-dedup",
            "--no-llm",
            "-r",
            "cli,json,html,markdown,sarif",
            "-o",
            str(output_dir),
            *extra,
        ],
    )


def test_baseline_fails_without_ignore_file(mismatched_skill: Path, tmp_path: Path) -> None:
    result = _invoke(mismatched_skill, tmp_path / "reports")

    assert result.exit_code == 1
    assert "name_consistency" in result.output


def test_explicit_ignore_file_removes_finding_from_gate(mismatched_skill: Path, tmp_path: Path) -> None:
    ignore_file = tmp_path / "known-issues.yaml"
    ignore_file.write_text(_IGNORE_DOCUMENT, encoding="utf-8")
    output_dir = tmp_path / "reports"

    result = _invoke(mismatched_skill, output_dir, "--ignore-file", str(ignore_file))

    assert result.exit_code == 0, result.output
    plain = strip_terminal_controls(result.output)
    assert "Suppressed Findings" in plain
    assert "rename-batch" in plain

    payload = _json_report(output_dir)
    assert payload["overall_passed"] is True
    ledger = payload["suppressions"]
    assert ledger["suppressed_count"] == 1
    assert ledger["source"] == str(ignore_file)
    assert ledger["discovered"] is False
    record = ledger["rules"][0]["findings"][0]
    assert record["check"] == "SCHEMA.name_consistency"
    assert record["matched_path"] == "SKILL.md"
    assert record["reason"] == "accepted for the tracked rename batch"
    suppressed = [item for item in payload["results"] if item.get("suppressed_findings")]
    assert len(suppressed) == 1
    # The suppressed finding no longer appears among active findings.
    assert all(
        finding["check_name"] != "name_consistency" for item in payload["results"] for finding in item["findings"]
    )

    sarif = json.loads(next(output_dir.glob("*.sarif.json")).read_text(encoding="utf-8"))
    run = sarif["runs"][0]
    assert run["properties"]["suppressedFindingCount"] == 1
    suppressed_results = [item for item in run["results"] if item.get("suppressions")]
    assert len(suppressed_results) == 1
    assert suppressed_results[0]["suppressions"][0]["properties"]["ruleId"] == "rename-batch"

    markdown = next(output_dir.glob("skillevaluator-output-*.md")).read_text(encoding="utf-8")
    assert "## Suppressed Findings" in markdown
    assert "rename-batch" in markdown

    html = next(output_dir.glob("*.html")).read_text(encoding="utf-8")
    assert "Suppressed findings" in html
    assert "rename-batch" in html


def test_ignore_file_is_auto_discovered_in_skill_root(mismatched_skill: Path, tmp_path: Path) -> None:
    (mismatched_skill / "skillevaluator-ignore.yaml").write_text(_IGNORE_DOCUMENT, encoding="utf-8")

    result = _invoke(mismatched_skill, tmp_path / "reports")

    assert result.exit_code == 0, result.output
    payload = _json_report(tmp_path / "reports")
    assert payload["suppressions"]["discovered"] is True
    assert payload["suppressions"]["suppressed_count"] == 1


def test_strict_disables_auto_discovery(mismatched_skill: Path, tmp_path: Path) -> None:
    (mismatched_skill / "skillevaluator-ignore.yaml").write_text(_IGNORE_DOCUMENT, encoding="utf-8")

    result = _invoke(mismatched_skill, tmp_path / "reports", "--strict")

    assert result.exit_code == 1
    assert "name_consistency" in result.output


def test_blanket_wildcard_ignore_file_fails_closed(mismatched_skill: Path, tmp_path: Path) -> None:
    ignore_file = tmp_path / "bad.yaml"
    ignore_file.write_text(
        "ignores:\n  - check: SCHEMA.name_consistency\n    paths: ['**']\n",
        encoding="utf-8",
    )

    result = _invoke(mismatched_skill, tmp_path / "reports", "--ignore-file", str(ignore_file))

    assert result.exit_code != 0
    assert "all-wildcard" in result.output


def test_unmatched_rule_is_reported_as_warning(mismatched_skill: Path, tmp_path: Path) -> None:
    ignore_file = tmp_path / "known-issues.yaml"
    ignore_file.write_text(
        """
ignores:
  - check: SCHEMA.frontmatter_format
    paths: ["docs/*.md"]
""",
        encoding="utf-8",
    )

    result = _invoke(mismatched_skill, tmp_path / "reports", "--ignore-file", str(ignore_file))

    assert result.exit_code == 1
    payload = _json_report(tmp_path / "reports")
    warnings = payload["suppressions"]["warnings"]
    assert any("never matched a finding" in warning for warning in warnings)


def test_expired_rule_does_not_suppress(mismatched_skill: Path, tmp_path: Path) -> None:
    ignore_file = tmp_path / "expired.yaml"
    ignore_file.write_text(
        """
ignores:
  - check: SCHEMA.name_consistency
    paths: [SKILL.md]
    expires: 2020-01-01
""",
        encoding="utf-8",
    )

    result = _invoke(mismatched_skill, tmp_path / "reports", "--ignore-file", str(ignore_file))

    assert result.exit_code == 1
    payload = _json_report(tmp_path / "reports")
    ledger = payload["suppressions"]
    assert ledger["suppressed_count"] == 0
    assert ledger["rules"][0]["status"] == "expired"
    assert any("expired" in warning for warning in ledger["warnings"])


def test_no_ignore_file_keeps_default_report_shape(mismatched_skill: Path, tmp_path: Path) -> None:
    result = _invoke(mismatched_skill, tmp_path / "reports")

    assert result.exit_code == 1
    payload = _json_report(tmp_path / "reports")
    assert "suppressions" not in payload


def test_catalog_argv_forwards_ignore_flags(tmp_path: Path) -> None:
    ignore_file = tmp_path / "known-issues.yaml"
    ignore_file.write_text(_IGNORE_DOCUMENT, encoding="utf-8")
    ctx = SimpleNamespace(
        params={
            "ignore_file": ignore_file,
            "strict": True,
            "llm": False,
            "dedup": True,
            "agent_eval": False,
            "autopilot": False,
            "min_score": 70,
            "env_mode": "docker",
            "report_formats": (),
            "include_skills": (),
            "environment_kwarg": (),
            "agent_model": (),
            "verbose": False,
            "full": False,
            "tiers": None,
            "checks": None,
            "previous_version": None,
            "fail_fast": False,
            "continue_on_failure": False,
            "llm_verify": False,
            "quality_config": None,
            "external": False,
            "policy_path": None,
            "profile": None,
            "block_on_dedup": None,
            "block_on_agent_eval": None,
            "agents": None,
            "skip_baseline": False,
            "n_concurrent": None,
            "max_agents": None,
            "n_attempts": None,
            "pass_threshold": None,
            "stop_on_pass": None,
            "model": None,
            "grading_mode": None,
            "results_dir": None,
            "copy_repo": False,
            "timeout_multiplier": None,
            "harbor_keep_jobs": False,
            "agent_runtime_preflight": None,
            "evaluated_source_repository": None,
            "evaluated_source_revision": None,
            "evaluator_container_revision": None,
        }
    )

    argv = _catalog_child_argv_from_ctx(ctx, tmp_path / "child-skill", tmp_path / "out")

    assert "--strict" in argv
    assert "--ignore-file" in argv
    assert str(ignore_file.resolve()) in argv
