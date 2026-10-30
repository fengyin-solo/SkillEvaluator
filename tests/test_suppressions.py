# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for reviewable finding exemptions (ignore file parsing and suppression)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.suppressions import (
    SUPPRESSION_LEDGER_KEY,
    IgnoreFileError,
    apply_ignore_rules,
    get_suppression_ledger,
    glob_matches,
    load_ignore_file,
    skill_relative_path,
    suppressed_finding_records,
)


def _write_ignore(tmp_path: Path, text: str, *, name: str = "skillevaluator-ignore.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_load_valid_document_normalizes_check_and_defaults(tmp_path: Path) -> None:
    path = _write_ignore(
        tmp_path,
        """
version: 1
ignores:
  - id: rename-batch
    check: schema.name_consistency
    paths:
      - SKILL.md
      - scripts/*.py
    reason: tracked in SEC-1234
    expires: 2030-01-01
""",
    )

    rules = load_ignore_file(path)

    assert len(rules) == 1
    rule = rules[0]
    assert rule.rule_id == "rename-batch"
    assert rule.category == "SCHEMA"
    assert rule.check_name == "name_consistency"
    assert rule.check_id == "SCHEMA.name_consistency"
    assert rule.paths == ("SKILL.md", "scripts/*.py")
    assert rule.reason == "tracked in SEC-1234"
    assert rule.expires.isoformat() == "2030-01-01"


def test_rule_id_defaults_to_position(tmp_path: Path) -> None:
    path = _write_ignore(
        tmp_path,
        """
ignores:
  - check: SCHEMA.a
    paths: [a.md]
  - check: SCHEMA.b
    paths: [b.md]
""",
    )

    rules = load_ignore_file(path)

    assert [rule.rule_id for rule in rules] == ["rule-1", "rule-2"]


@pytest.mark.parametrize(
    "document",
    [
        "ignores: []\n",
        "{}\n",
        "version: 2\nignores: []\n",
        "ignores:\n  - paths: [SKILL.md]\n",
        "ignores:\n  - check: SCHEMA\n    paths: [SKILL.md]\n",
        "ignores:\n  - check: SCHEMA.*\n    paths: [SKILL.md]\n",
        "ignores:\n  - check: '*'\n    paths: [SKILL.md]\n",
        "ignores:\n  - check: SCHEMA.a\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: []\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: nope\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: [SKILL.md]\n    bogus: 1\n",
        "top: 1\nignores:\n  - check: SCHEMA.a\n    paths: [SKILL.md]\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: [SKILL.md]\n    id: 'bad id'\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: [SKILL.md]\n    id: dup\n  - check: SCHEMA.b\n    paths: [b.md]\n    id: dup\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: [SKILL.md]\n    expires: 'not-a-date'\n",
        "ignores:\n  - check: SCHEMA.a\n    paths: [SKILL.md]\n    reason: 12\n",
    ],
)
def test_invalid_documents_raise(tmp_path: Path, document: str) -> None:
    path = _write_ignore(tmp_path, document)

    with pytest.raises(IgnoreFileError):
        load_ignore_file(path)


@pytest.mark.parametrize(
    "glob",
    [
        "*",
        "**",
        "**/*",
        "*/**",
        "???",
        "/SKILL.md",
        "C:/SKILL.md",
        "../SKILL.md",
        "scripts/../SKILL.md",
        "scripts/",
        "scripts//a.py",
        "a/./b.py",
        "[abc",
        "a[b",
        "scripts/**.py",
        "a\\b.py",
    ],
)
def test_unsafe_or_blanket_globs_are_rejected(tmp_path: Path, glob: str) -> None:
    path = _write_ignore(
        tmp_path,
        f"""
ignores:
  - check: SCHEMA.a
    paths: [{glob!r}]
""",
    )

    with pytest.raises(IgnoreFileError):
        load_ignore_file(path)


def test_scoped_globs_are_accepted(tmp_path: Path) -> None:
    path = _write_ignore(
        tmp_path,
        """
ignores:
  - check: SCHEMA.a
    paths:
      - SKILL.md
      - scripts/*.py
      - "**/*.md"
      - references/docs-?.md
      - src/[ab].py
""",
    )

    rules = load_ignore_file(path)

    assert rules[0].paths == ("SKILL.md", "scripts/*.py", "**/*.md", "references/docs-?.md", "src/[ab].py")


def test_invalid_yaml_and_duplicate_keys_raise(tmp_path: Path) -> None:
    path = _write_ignore(
        tmp_path,
        """
ignores:
  - check: SCHEMA.a
    paths: [a.md]
    paths: [b.md]
""",
    )

    with pytest.raises(IgnoreFileError):
        load_ignore_file(path)


# ---------------------------------------------------------------------------
# Skill-relative path handling and glob matching
# ---------------------------------------------------------------------------


def test_skill_relative_path_handles_relative_and_absolute() -> None:
    root = Path("/repo/skills/my-skill")

    assert skill_relative_path("SKILL.md", root) == "SKILL.md"
    assert skill_relative_path("./SKILL.md", root) == "SKILL.md"
    assert skill_relative_path("scripts/a.py", root) == "scripts/a.py"
    assert skill_relative_path("/repo/skills/my-skill/SKILL.md", root) == "SKILL.md"
    assert skill_relative_path("scripts\\a.py", root) == "scripts/a.py"


def test_skill_relative_path_rejects_escape_and_drive() -> None:
    root = Path("/repo/skills/my-skill")

    assert skill_relative_path("../other/SKILL.md", root) is None
    assert skill_relative_path("/repo/other/SKILL.md", root) is None
    assert skill_relative_path("a/../b", root) is None
    assert skill_relative_path("C:/repo/SKILL.md", root) is None
    assert skill_relative_path("", root) is None
    assert skill_relative_path(None, root) is None


def test_glob_matching_is_segment_aware() -> None:
    assert glob_matches("scripts/*.py", "scripts/a.py")
    assert not glob_matches("scripts/*.py", "scripts/sub/a.py")
    assert glob_matches("**/*.py", "a.py")
    assert glob_matches("**/*.py", "scripts/sub/a.py")
    assert not glob_matches("**/*.py", "scripts/a.txt")
    assert glob_matches("refs/**/*.md", "refs/a/b/c.md")
    assert glob_matches("SKILL.md", "SKILL.md")
    assert not glob_matches("SKILL.md", "README.md")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def _finding_result() -> ValidationResult:
    result = ValidationResult(validator_name="Schema Validation", validator_description="schema")
    result.add_structured_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="name_consistency",
            message="name mismatch",
            file_path="/repo/skills/my-skill/SKILL.md",
            line_number=1,
        ),
        is_error=True,
    )
    return result


def _rules(tmp_path: Path, text: str) -> list:
    return load_ignore_file(_write_ignore(tmp_path, text, name="known-issues.yaml"))


def test_matched_finding_is_suppressed_and_recorded(tmp_path: Path) -> None:
    result = _finding_result()
    assert not result.passed
    rules = _rules(
        tmp_path,
        """
ignores:
  - id: rename-batch
    check: SCHEMA.name_consistency
    paths: [SKILL.md]
    reason: accepted for the rename batch
    expires: 2030-01-01
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.passed
    assert result.findings == []
    assert ledger["suppressed_count"] == 1
    records = suppressed_finding_records(result)
    assert len(records) == 1
    record = records[0]
    assert record["rule_id"] == "rename-batch"
    assert record["check"] == "SCHEMA.name_consistency"
    assert record["matched_path"] == "SKILL.md"
    assert record["matched_glob"] == "SKILL.md"
    assert record["reason"] == "accepted for the rename batch"
    assert record["expires"] == "2030-01-01"
    assert record["source"].endswith("known-issues.yaml")
    assert get_suppression_ledger([result]) is ledger
    assert result.metadata[SUPPRESSION_LEDGER_KEY]["suppressed_count"] == 1


def test_unmatched_rule_is_warned_and_does_not_suppress(tmp_path: Path) -> None:
    result = _finding_result()
    rules = _rules(
        tmp_path,
        """
ignores:
  - check: SCHEMA.frontmatter_format
    paths: [docs/*.md]
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert not result.passed
    assert len(result.findings) == 1
    assert ledger["suppressed_count"] == 0
    assert "never matched a finding" in ledger["warnings"][0]
    assert ledger["rules"][0]["matched_count"] == 0


def test_expired_rule_does_not_suppress_and_warns(tmp_path: Path) -> None:
    result = _finding_result()
    rules = _rules(
        tmp_path,
        """
ignores:
  - check: SCHEMA.name_consistency
    paths: [SKILL.md]
    expires: 2020-12-31
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert not result.passed
    assert len(result.findings) == 1
    assert ledger["rules"][0]["status"] == "expired"
    assert ledger["expired_rule_count"] == 1
    assert any("expired" in warning for warning in ledger["warnings"])


def test_secret_findings_cannot_be_exempted(tmp_path: Path) -> None:
    result = ValidationResult(validator_name="Secrets Detection")
    result.add_structured_finding(
        Finding(
            category="SECRET",
            severity=Severity.CRITICAL,
            check_name="aws-access-key-id",
            message="hardcoded key",
            file_path="scripts/app.py",
            line_number=3,
        ),
        is_error=True,
    )
    rules = _rules(
        tmp_path,
        """
ignores:
  - check: SECRET.aws-access-key-id
    paths:
      - "scripts/*.py"
    reason: should never be accepted
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert not result.passed
    assert len(result.findings) == 1
    assert ledger["suppressed_count"] == 0
    assert ledger["blocked"][0]["reason"] == "secret findings cannot be exempted"
    assert any("secret findings cannot be exempted" in warning for warning in ledger["warnings"])
    # A blocked rule already has a specific warning; it must not also be
    # reported as "never matched".
    assert not any("never matched a finding" in warning for warning in ledger["warnings"])


def test_findings_from_incomplete_scans_cannot_be_exempted(tmp_path: Path) -> None:
    result = _finding_result()
    result.mark_scan_incomplete("semgrep")
    result.recalculate_from_findings()
    rules = _rules(
        tmp_path,
        """
ignores:
  - check: SCHEMA.name_consistency
    paths: [SKILL.md]
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert len(result.findings) == 1
    assert ledger["suppressed_count"] == 0
    assert "incomplete" in ledger["blocked"][0]["reason"]
    assert result.is_incomplete


def test_first_matching_rule_wins(tmp_path: Path) -> None:
    result = _finding_result()
    rules = _rules(
        tmp_path,
        """
ignores:
  - id: first
    check: SCHEMA.name_consistency
    paths: ["*.md"]
  - id: second
    check: SCHEMA.name_consistency
    paths: [SKILL.md]
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert ledger["suppressed_count"] == 1
    records = suppressed_finding_records(result)
    assert records[0]["rule_id"] == "first"
    assert ledger["rules"][1]["matched_count"] == 0


def test_path_scope_must_match_the_finding_location(tmp_path: Path) -> None:
    result = _finding_result()
    rules = _rules(
        tmp_path,
        """
ignores:
  - check: SCHEMA.name_consistency
    paths:
      - docs/SKILL.md
      - "references/*.md"
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert not result.passed
    assert len(result.findings) == 1
    assert ledger["suppressed_count"] == 0


def test_medium_advisory_finding_is_suppressed_and_counted(tmp_path: Path) -> None:
    result = ValidationResult(validator_name="Schema Validation")
    result.add_structured_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.MEDIUM,
            check_name="line_count",
            message="too many lines",
            file_path="SKILL.md",
            line_number=520,
        ),
        is_error=False,
    )
    rules = _rules(
        tmp_path,
        """
ignores:
  - check: SCHEMA.line_count
    paths: [SKILL.md]
    reason: split scheduled
""",
    )

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("known-issues.yaml"),
        discovered=False,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert result.findings == []
    assert result.passed
    assert ledger["suppressed_count"] == 1
    assert suppressed_finding_records(result)[0]["severity"] == "medium"


def test_discovered_flag_is_recorded(tmp_path: Path) -> None:
    result = _finding_result()
    rules = _rules(tmp_path, "ignores:\n  - check: SCHEMA.name_consistency\n    paths: [SKILL.md]\n")

    ledger = apply_ignore_rules(
        [result],
        rules,
        source=Path("/repo/skills/my-skill/skillevaluator-ignore.yaml"),
        discovered=True,
        skill_root=Path("/repo/skills/my-skill"),
    )

    assert ledger["discovered"] is True
    assert ledger["source"].endswith("skillevaluator-ignore.yaml")
