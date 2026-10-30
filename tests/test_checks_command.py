# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the read-only ``checks`` catalog command and its shared registry.

The catalog is the single source of truth reused by the Tier 1 runtime, so
these tests pin both the user-facing contract (stable JSON/text, filtering,
unknown reasons, offline availability) and the anti-drift invariants between
the registry and ``tier1.commands``.
"""

from __future__ import annotations

import builtins
import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from skillevaluator import check_catalog
from skillevaluator.check_catalog import (
    CHECK_REGISTRY,
    build_check_catalog,
    explain_unknown_content_type,
    run_checks_command,
)
from skillevaluator.cli import cli
from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
)
from skillevaluator.tier1 import commands as tier1_commands

REQUIRED_CATEGORY_IDS = [
    "schema",
    "version",
    "security",
    "pii",
    "license",
    "code-integrity",
    "dependency",
    "unicode",
    "quality",
    "lint",
]

CATALOG_TOP_LEVEL_KEYS = [
    "schema_version",
    "command",
    "tier",
    "content_type",
    "content_type_source",
    "content_type_reason",
    "target_path",
    "availability_probe",
    "checks",
]

CHECK_KEYS = [
    "id",
    "name",
    "description",
    "enabled_by_default",
    "activation",
    "applicable_content_types",
    "applies_to_unknown",
    "evidence_categories",
    "components",
    "requirements",
]


def _by_id(catalog: dict) -> dict[str, dict]:
    return {check["id"]: check for check in catalog["checks"]}


# ---------------------------------------------------------------------------
# Registry contents and runtime anti-drift invariants
# ---------------------------------------------------------------------------


def test_registry_covers_every_required_category_once_in_pipeline_order() -> None:
    assert [spec.id for spec in CHECK_REGISTRY] == REQUIRED_CATEGORY_IDS


def test_dependency_is_the_only_opt_in_check() -> None:
    dep = check_catalog.CHECK_SPECS["dependency"]
    assert dep.activation == check_catalog.ACTIVATION_OPT_IN
    assert dep.enabled_by_default is False
    assert [spec.id for spec in CHECK_REGISTRY if spec.activation == check_catalog.ACTIVATION_OPT_IN] == [
        "dependency"
    ]


def test_runtime_lineup_constants_derive_from_registry() -> None:
    # The runtime must not keep a second handwritten list that can drift.
    assert tuple(check_catalog.DEFAULT_CHECKS) == tuple(tier1_commands.DEFAULT_CHECKS)
    assert tuple(check_catalog.OPTIONAL_CHECKS) == tuple(tier1_commands.OPTIONAL_CHECKS)
    assert check_catalog.RECOGNIZED_CHECKS == tier1_commands.RECOGNIZED_CHECKS
    assert set(tier1_commands.DEFAULT_CHECKS) == set(REQUIRED_CATEGORY_IDS) - {"dependency"}
    assert tier1_commands.enabled_check_lineup(None) == list(tier1_commands.DEFAULT_CHECKS)
    assert tier1_commands.enabled_check_lineup("deps") == ["dependency"]


def test_skill_only_checks_match_runtime_gate() -> None:
    skill_only = {spec.id for spec in CHECK_REGISTRY if spec.skill_only}
    assert skill_only == {"version", "quality", "lint"}
    # Applicability must agree with the gate run_validation applies.
    for spec in CHECK_REGISTRY:
        assert spec.applies_to(CONTENT_TYPE_SKILL) is True
        assert spec.applies_to(CONTENT_TYPE_UNKNOWN) is True  # unknown -> skill fallback
        if spec.skill_only:
            for content_type in (CONTENT_TYPE_RULES, CONTENT_TYPE_WORKFLOWS, CONTENT_TYPE_PLUGIN):
                assert spec.applies_to(content_type) is False
        else:
            for content_type in (CONTENT_TYPE_RULES, CONTENT_TYPE_WORKFLOWS, CONTENT_TYPE_PLUGIN):
                assert spec.applies_to(content_type) is True


def test_every_check_carries_evidence_and_a_live_component_name() -> None:
    for spec in CHECK_REGISTRY:
        assert spec.evidence_categories, spec.id
        names = spec.component_names()
        assert names and all(isinstance(name, str) and name for name in names), spec.id


def test_schema_validator_selection_matches_runtime_helper() -> None:
    assert type(check_catalog.schema_validator_for(CONTENT_TYPE_RULES)).__name__ == "RulesSchemaValidator"
    assert type(check_catalog.schema_validator_for(CONTENT_TYPE_WORKFLOWS)).__name__ == "WorkflowsSchemaValidator"
    assert type(check_catalog.schema_validator_for(CONTENT_TYPE_PLUGIN)).__name__ == "PluginSchemaValidator"
    assert type(check_catalog.schema_validator_for(CONTENT_TYPE_SKILL)).__name__ == "SchemaValidator"
    assert type(check_catalog.schema_validator_for(CONTENT_TYPE_UNKNOWN)).__name__ == "SchemaValidator"


# ---------------------------------------------------------------------------
# JSON contract
# ---------------------------------------------------------------------------


def test_unfiltered_json_has_stable_top_level_and_check_key_order() -> None:
    catalog = build_check_catalog()
    assert list(catalog.keys()) == CATALOG_TOP_LEVEL_KEYS
    assert catalog["content_type"] is None
    assert catalog["content_type_source"] is None
    assert catalog["target_path"] is None
    assert catalog["availability_probe"] == "offline"
    assert catalog["schema_version"] == 1
    assert [check["id"] for check in catalog["checks"]] == REQUIRED_CATEGORY_IDS
    for check in catalog["checks"]:
        assert list(check.keys()) == CHECK_KEYS
        # No filter -> no per-check applicability verdict is implied.
        assert "applicable" not in check


def test_dependency_metadata_marks_external_scanners() -> None:
    check = _by_id(build_check_catalog())["dependency"]
    assert check["enabled_by_default"] is False
    assert check["activation"] == "opt-in"
    assert check["evidence_categories"] == ["CVE", "SAFETY"]
    requirements = {req["name"] + (f":{req['enabled_by']}" if req["enabled_by"] else ""): req for req in check["requirements"]}
    pip_audit = requirements["pip-audit"]
    assert pip_audit["kind"] == "executable"
    assert pip_audit["required"] is True
    assert pip_audit["bundled_extra"] == "security"
    assert pip_audit["probe"] == "path"
    assert "available" in pip_audit
    safety = requirements["safety"]
    assert safety["required"] is False and safety["bundled_extra"] is None


def test_code_integrity_and_security_requirements_are_offline_documented() -> None:
    by_id = _by_id(build_check_catalog())
    ci_tools = {req["name"]: req for req in by_id["code-integrity"]["requirements"]}
    assert set(ci_tools) == {"bandit", "semgrep", "gitleaks"}
    assert ci_tools["bandit"]["bundled_extra"] == "security"
    assert ci_tools["semgrep"]["bundled_extra"] is None
    assert ci_tools["gitleaks"]["bundled_extra"] is None

    security_reqs = by_id["security"]["requirements"]
    skillspector = next(req for req in security_reqs if req["name"] == "skillspector")
    assert skillspector["kind"] == "executable" and skillspector["required"] is True
    llm_reqs = [req for req in security_reqs if req["kind"] == "package"]
    assert {req["enabled_by"] for req in llm_reqs} == {"--llm", "--llm-verify"}
    for req in llm_reqs:
        assert req["probe"] == "import" and req["bundled_extra"] == "llm"
        assert "installed" in req


# ---------------------------------------------------------------------------
# Filtering and content-type resolution
# ---------------------------------------------------------------------------


def test_type_flag_resolves_applicability_and_schema_component() -> None:
    catalog = build_check_catalog(content_type=CONTENT_TYPE_RULES)
    assert catalog["content_type"] == "rules"
    assert catalog["content_type_source"] == "flag"
    by_id = _by_id(catalog)
    for check_id in ("version", "quality", "lint"):
        assert by_id[check_id]["applicable"] is False
    for check_id in ("schema", "security", "pii", "license", "code-integrity", "dependency", "unicode"):
        assert by_id[check_id]["applicable"] is True
    assert by_id["schema"]["components"] == ["Rules Schema & Repository Governance"]


def test_only_applicable_drops_skill_only_checks_for_rules() -> None:
    catalog = build_check_catalog(content_type=CONTENT_TYPE_RULES, only_applicable=True)
    ids = [check["id"] for check in catalog["checks"]]
    assert ids == ["schema", "security", "pii", "license", "code-integrity", "dependency", "unicode"]


def test_path_auto_detects_skill_and_applies_everything(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# skill")
    catalog = build_check_catalog(target_path=tmp_path)
    assert catalog["content_type"] == CONTENT_TYPE_SKILL
    assert catalog["content_type_source"] == "auto"
    assert catalog["content_type_reason"] is None
    assert all(check["applicable"] for check in catalog["checks"])
    schema = _by_id(catalog)["schema"]
    assert schema["components"] == ["Schema & Repository Governance"]


def test_unknown_directory_explains_why(tmp_path: Path) -> None:
    reason = explain_unknown_content_type(tmp_path)
    assert "SKILL.md" in reason
    catalog = build_check_catalog(target_path=tmp_path)
    assert catalog["content_type"] == CONTENT_TYPE_UNKNOWN
    assert catalog["content_type_source"] == "auto"
    assert catalog["content_type_reason"] == reason
    # Unknown uses the skill fallback, so skill-only checks still apply.
    by_id = _by_id(catalog)
    assert by_id["quality"]["applicable"] is True


def test_nonexistent_target_reports_unknown_reason(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    catalog = build_check_catalog(target_path=missing)
    assert catalog["content_type"] == CONTENT_TYPE_UNKNOWN
    assert "does not exist" in catalog["content_type_reason"]


def test_flag_overrides_path_detection(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("# skill")
    catalog = build_check_catalog(content_type=CONTENT_TYPE_PLUGIN, target_path=tmp_path)
    assert catalog["content_type"] == CONTENT_TYPE_PLUGIN
    assert catalog["content_type_source"] == "flag"
    assert _by_id(catalog)["schema"]["components"] == ["Plugin Schema & Bundle References"]


# ---------------------------------------------------------------------------
# Offline / read-only guarantees
# ---------------------------------------------------------------------------


def test_building_catalog_never_runs_scanners_or_imports_llm_clients(monkeypatch) -> None:
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("catalog build must not execute an external process")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)

    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"openai", "litellm", "anthropic"}:
            raise AssertionError(f"catalog build must not import optional LLM client {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)

    # Must succeed repeatedly for every filter without launching a scanner.
    build_check_catalog()
    build_check_catalog(content_type=CONTENT_TYPE_RULES)
    build_check_catalog(target_path=Path("/nonexistent/path"))


def test_executable_availability_reflects_path_lookup_without_running(monkeypatch) -> None:
    from skillevaluator.utils import tool_runner

    # Force is_available True/False via the PATH-only property; run is untouched.
    monkeypatch.setattr(tool_runner.Tools.semgrep, "_path", "/fake/semgrep", raising=False)
    catalog = build_check_catalog(content_type=CONTENT_TYPE_SKILL)
    semgrep = next(
        req
        for req in _by_id(catalog)["code-integrity"]["requirements"]
        if req["name"] == "semgrep"
    )
    assert semgrep["available"] is True

    monkeypatch.setattr(tool_runner.Tools.semgrep, "_path", None, raising=False)
    catalog = build_check_catalog(content_type=CONTENT_TYPE_SKILL)
    semgrep = next(
        req
        for req in _by_id(catalog)["code-integrity"]["requirements"]
        if req["name"] == "semgrep"
    )
    assert semgrep["available"] is False


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_checks_json_is_valid_and_read_only() -> None:
    result = CliRunner().invoke(cli, ["checks", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert list(payload.keys()) == CATALOG_TOP_LEVEL_KEYS
    assert [check["id"] for check in payload["checks"]] == REQUIRED_CATEGORY_IDS


def test_cli_checks_text_marks_dependency_opt_in() -> None:
    result = CliRunner().invoke(cli, ["checks"])
    assert result.exit_code == 0
    assert "dependency" in result.output
    assert "opt-in" in result.output
    assert "offline" in result.output


def test_cli_checks_type_filter_and_only_applicable() -> None:
    result = CliRunner().invoke(cli, ["checks", "--type", "workflows", "--only-applicable", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content_type"] == "workflows"
    ids = [check["id"] for check in payload["checks"]]
    assert "version" not in ids and "quality" not in ids and "lint" not in ids
    assert "schema" in ids


def test_cli_checks_unknown_path_sets_reason(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["checks", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content_type"] == CONTENT_TYPE_UNKNOWN
    assert payload["content_type_reason"]


def test_cli_checks_rejects_unknown_type() -> None:
    result = CliRunner().invoke(cli, ["checks", "--type", "bogus"])
    assert result.exit_code != 0


def test_run_checks_command_returns_zero(capsys) -> None:
    code = run_checks_command(target_path=None, content_type=None, as_json=True, only_applicable=False)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "checks"
