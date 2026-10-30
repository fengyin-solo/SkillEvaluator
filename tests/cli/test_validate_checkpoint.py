# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI tests for ``validate --checkpoint-dir`` (resumable catalog validation)."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.cli import cli
from skillevaluator.tier1.commands import run_validation as real_run_validation
from skillevaluator.validation_checkpoint import record_filename

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "skills" / "simple"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
COUNTS = re.compile(
    r"Checkpoint resume: (\d+) reused, (\d+) re-executed, (\d+) newly executed"
)


def _plain(text: str) -> str:
    return ANSI.sub("", text)


def _make_catalog(root: Path, names: tuple[str, ...] = ("simple", "simple2")) -> Path:
    catalog = root / "catalog"
    for name in names:
        shutil.copytree(FIXTURE, catalog / name)
        manifest = catalog / name / "SKILL.md"
        text = manifest.read_text(encoding="utf-8").replace("name: simple", f"name: {name}")
        manifest.write_text(text, encoding="utf-8")
    return catalog


def _base_argv(catalog: Path, out: str, cp: str | None, *, report: str = "json", **extra: object) -> list[str]:
    argv = [
        "validate",
        str(catalog),
        "--no-tier3",
        "--no-llm",
        "--no-dedup",
        "--checks",
        str(extra.pop("checks", "schema")),
        "-r",
        report,
        "-o",
        out,
    ]
    if cp is not None:
        argv.extend(["--checkpoint-dir", cp])
    for flag, value in extra.items():
        flag_arg = f"--{flag.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag_arg)
        else:
            argv.extend([flag_arg, str(value)])
    return argv


def _run(argv: list[str]) -> object:
    return CliRunner().invoke(cli, argv)


def _counts(output: str) -> tuple[int, int, int]:
    match = COUNTS.search(_plain(output))
    assert match, output
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _count_validations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def counting(target: Path, **kwargs: object) -> list:
        calls.append(str(target))
        return real_run_validation(target, **kwargs)

    monkeypatch.setattr(cli_module, "run_validation", counting)
    return calls


def _summary(out_dir: str) -> dict:
    return json.loads(Path(out_dir, "catalog-summary.json").read_text(encoding="utf-8"))


def _skill_json(out_dir: str, name: str) -> dict:
    files = list(Path(out_dir, name).glob("skillevaluator-output-*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def _semantic_summary(summary: dict) -> dict:
    return {
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "overall_passed": summary["overall_passed"],
        "severity_totals": summary["severity_totals"],
        "skills": sorted(
            (
                {
                    "name": skill["name"],
                    "passed": skill["passed"],
                    "overall_passed": skill.get("overall_passed"),
                    "overall_status": skill.get("overall_status"),
                    "severity_counts": skill.get("severity_counts"),
                }
                for skill in summary["skills"]
            ),
            key=lambda item: item["name"],
        ),
    }


# ---------------------------------------------------------------------------
# Fresh run + resume
# ---------------------------------------------------------------------------


def test_serial_catalog_checkpoint_then_resume_reuses_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd())

        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output
        assert _counts(first.output) == (0, 0, 2)
        assert len(calls) == 2

        # Same command, separate report dir: nothing is re-validated.
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        plain = _plain(second.output)
        assert "checkpoint reuse: skill 1/2 · simple" in plain
        assert "checkpoint reuse: skill 2/2 · simple2" in plain
        assert _counts(second.output) == (2, 0, 0)
        assert len(calls) == 2  # no additional run_validation calls

        # JSON reports are restored from the checkpoint.
        for name in ("simple", "simple2"):
            assert list(Path("out2", name).glob("skillevaluator-output-*.json"))

        # Final summary and JSON results match an uninterrupted run.
        summary1, summary2 = _summary("out1"), _summary("out2")
        assert _semantic_summary(summary1) == _semantic_summary(summary2)
        for name in ("simple", "simple2"):
            payload1 = _skill_json("out1", name)
            payload2 = _skill_json("out2", name)
            assert payload1["overall_passed"] == payload2["overall_passed"] is True
            assert payload1["severity_counts"] == payload2["severity_counts"]
            assert payload1["results"] and len(payload1["results"]) == len(payload2["results"])
        # Per-skill records bind the exact same JSON basename.
        assert summary1["skills"][0]["json_report"] == summary2["skills"][0]["json_report"]


def test_resume_reexecutes_changed_skill_and_reuses_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output
        assert len(calls) == 2

        (catalog / "simple2" / "SKILL.md").write_text(
            (catalog / "simple2" / "SKILL.md").read_text(encoding="utf-8") + "\n## Changed\nNew body line.\n",
            encoding="utf-8",
        )
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        reused, reexecuted, newly = _counts(second.output)
        assert (reused, reexecuted, newly) == (1, 1, 0)
        assert "plan changed (skill content)" in _plain(second.output)
        assert len(calls) == 3
        assert calls[-1].endswith("simple2")


# ---------------------------------------------------------------------------
# Plan identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("checks", "run2_extra", "phrase"),
    [
        ("schema", ["--checks", "schema,version"], "check set"),
        ("schema,quality", ["--min-score", "30"], "validation options"),
        ("schema", ["--llm"], "validation options"),
    ],
)
def test_plan_option_change_forces_reexecution(
    monkeypatch: pytest.MonkeyPatch, checks: str, run2_extra: list[str], phrase: str
) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp", checks=checks))
        assert first.exit_code == 0, first.output
        second = runner.invoke(cli, [*_base_argv(catalog, "out2", "cp", checks=checks), *run2_extra])
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (0, 2, 0)
        assert phrase in _plain(second.output)


def test_quality_config_change_invalidates_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        quality = Path("quality.yaml")
        quality.write_text(
            "expected_headers:\n"
            '  - heading: "## Prerequisites"\n'
            '    alternatives: ["## Requirements"]\n'
            "    points: 5\n"
            '    message: "missing prerequisites"\n'
            '    suggestion: "document setup"\n',
            encoding="utf-8",
        )
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp", checks="schema,quality"))
        assert first.exit_code == 0, first.output
        second = runner.invoke(
            cli, _base_argv(catalog, "out2", "cp", checks="schema,quality", quality_config=str(quality))
        )
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (0, 2, 0)
        assert "quality config" in _plain(second.output)


def test_policy_file_change_invalidates_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        policy = Path("policy.yaml")
        policy.write_text(
            "profile: external\nidentity: {}\nseverity_overrides: {}\n",
            encoding="utf-8",
        )
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp", policy=str(policy)))
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (0, 2, 0)
        assert "policy file" in _plain(second.output)


def test_report_format_change_invalidates_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp", report="json"))
        assert first.exit_code == 0, first.output
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp", report="cli"))
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (0, 2, 0)
        assert "report settings" in _plain(second.output)


def test_tool_version_change_invalidates_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skillevaluator.validation_checkpoint as checkpoint_module

    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output
        monkeypatch.setattr(checkpoint_module, "__version__", "9.9.9-future")
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (0, 2, 0)
        assert "tool version" in _plain(second.output)


# ---------------------------------------------------------------------------
# New / deleted skills
# ---------------------------------------------------------------------------


def test_new_skill_runs_and_deleted_skill_record_stays_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple", "simple2"))
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output
        records_after_first = sorted(p.name for p in Path("cp").glob("skill-*.json"))
        assert len(records_after_first) == 2

        shutil.copytree(FIXTURE, catalog / "gamma")
        (catalog / "gamma" / "SKILL.md").write_text(
            "---\nname: gamma\n"
            "description: Gamma added later.\n"
            "metadata:\n  author: Gamma Author <gamma@example.com>\n"
            "---\n# Gamma\n",
            encoding="utf-8",
        )
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (2, 0, 1)
        assert _summary("out2")["total"] == 3
        assert len(list(Path("cp").glob("skill-*.json"))) == 3

        # Deleting gamma leaves its record on disk as history but this run's
        # summary only counts the remaining skills, all reused.
        shutil.rmtree(catalog / "gamma")
        third = runner.invoke(cli, _base_argv(catalog, "out3", "cp"))
        assert third.exit_code == 0, third.output
        assert _counts(third.output) == (2, 0, 0)
        assert _summary("out3")["total"] == 2
        assert len(list(Path("cp").glob("skill-*.json"))) == 3  # history retained


# ---------------------------------------------------------------------------
# Corruption / incompatibility / foreign targets
# ---------------------------------------------------------------------------


def test_corrupt_record_warns_reruns_and_is_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple",))
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output

        record = Path("cp") / record_filename(catalog.resolve(), "simple")
        record.write_text("{broken json", encoding="utf-8")
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        plain = _plain(second.output)
        assert "valid JSON" in plain
        assert _counts(second.output) == (0, 1, 0)

        third = runner.invoke(cli, _base_argv(catalog, "out3", "cp"))
        assert third.exit_code == 0, third.output
        assert _counts(third.output) == (1, 0, 0)


def test_incompatible_checkpoint_version_warns_and_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple",))
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code == 0, first.output

        record = Path("cp") / record_filename(catalog.resolve(), "simple")
        payload = json.loads(record.read_text(encoding="utf-8"))
        payload["checkpoint_version"] = 999
        record.write_text(json.dumps(payload), encoding="utf-8")

        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        plain = _plain(second.output)
        assert "incompatible checkpoint format version 999" in plain
        assert _counts(second.output) == (0, 1, 0)


def test_foreign_target_records_warn_and_are_not_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog_a = _make_catalog(Path("a"))
        catalog_b = _make_catalog(Path("b"), ("simple",))
        first = runner.invoke(cli, _base_argv(catalog_a, "out-a", "cp"))
        assert first.exit_code == 0, first.output

        second = runner.invoke(cli, _base_argv(catalog_b, "out-b", "cp"))
        assert second.exit_code == 0, second.output
        plain = _plain(second.output)
        assert "different validation target" in plain
        assert _counts(second.output) == (0, 0, 1)


# ---------------------------------------------------------------------------
# Failures and incomplete results
# ---------------------------------------------------------------------------


def test_failed_skill_is_not_cached_then_passes_after_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("good", "bad"))
        bad_manifest = catalog / "bad" / "SKILL.md"
        bad_manifest.write_text(
            bad_manifest.read_text(encoding="utf-8").replace("name: bad", "name: Bad_Name"),
            encoding="utf-8",
        )
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp"))
        assert first.exit_code != 0
        good_record = Path("cp") / record_filename(catalog.resolve(), "good")
        bad_record = Path("cp") / record_filename(catalog.resolve(), "bad")
        assert good_record.is_file()
        assert not bad_record.exists()

        bad_manifest.write_text(
            bad_manifest.read_text(encoding="utf-8").replace("name: Bad_Name", "name: bad"),
            encoding="utf-8",
        )
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp"))
        assert second.exit_code == 0, second.output
        assert bad_record.is_file()
        assert _counts(second.output) == (1, 0, 1)
        assert _summary("out2")["overall_passed"] is True


# ---------------------------------------------------------------------------
# Parallel workers
# ---------------------------------------------------------------------------


def test_parallel_checkpoint_resume_restores_json_and_summary() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd())
        first = runner.invoke(cli, [*_base_argv(catalog, "out1", "cp"), "--workers", "2"])
        assert first.exit_code == 0, first.output
        assert "Validating 2 skills with 2 workers" in _plain(first.output)

        second = runner.invoke(cli, [*_base_argv(catalog, "out2", "cp"), "--workers", "2"])
        assert second.exit_code == 0, second.output
        plain = _plain(second.output)
        assert "parallel catalog mode" not in plain  # executor fully skipped
        assert _counts(second.output) == (2, 0, 0)
        for name in ("simple", "simple2"):
            assert list(Path("out2", name).glob("skillevaluator-output-*.json"))
        assert _semantic_summary(_summary("out1")) == _semantic_summary(_summary("out2"))


# ---------------------------------------------------------------------------
# Scope and safety
# ---------------------------------------------------------------------------


def test_single_skill_ignores_checkpoint_dir() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        skill = Path("simple")
        shutil.copytree(FIXTURE, skill)
        result = runner.invoke(
            cli,
            [
                "validate",
                str(skill),
                "--no-tier3",
                "--no-llm",
                "--no-dedup",
                "--checks",
                "schema",
                "-o",
                "out",
                "--checkpoint-dir",
                "cp",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not Path("cp").exists()


def test_cli_only_records_round_trip_without_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple",))
        first = runner.invoke(cli, _base_argv(catalog, "out1", "cp", report="cli"))
        assert first.exit_code == 0, first.output
        assert list(Path("cp").glob("skill-*.json"))
        second = runner.invoke(cli, _base_argv(catalog, "out2", "cp", report="cli"))
        assert second.exit_code == 0, second.output
        assert _counts(second.output) == (1, 0, 0)
        assert not list(Path("out2", "simple").glob("*.json"))
        assert "json_report" not in _summary("out2")["skills"][0]


def test_checkpoint_dir_inside_skill_is_rejected() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple",))
        result = runner.invoke(cli, _base_argv(catalog, "out", str(catalog / "simple" / ".cp")))
        assert result.exit_code != 0
        assert "must not be inside a skill" in _plain(result.output)
        assert not (catalog / "simple" / ".cp").exists()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses permission bits")
def test_unwritable_checkpoint_dir_fails_before_scanning(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_validations(monkeypatch)
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple",))
        readonly = Path("readonly")
        readonly.mkdir(mode=0o500)
        try:
            result = runner.invoke(cli, _base_argv(catalog, "out", str(readonly / "cp")))
        finally:
            readonly.chmod(0o700)
        assert result.exit_code != 0
        plain = _plain(result.output)
        assert "checkpoint" in plain.lower()
        assert calls == []
        assert not Path("out", "catalog-summary.json").exists()


def test_checkpoint_dir_symlink_leaf_is_rejected() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        catalog = _make_catalog(Path.cwd(), ("simple",))
        Path("real").mkdir()
        Path("cp-link").symlink_to(Path("real"), target_is_directory=True)
        result = runner.invoke(cli, _base_argv(catalog, "out", "cp-link"))
        assert result.exit_code != 0
        assert "symlink" in _plain(result.output)
