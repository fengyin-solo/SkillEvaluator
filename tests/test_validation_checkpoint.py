# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for resumable catalog validation checkpoints."""

from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skillevaluator import validation_checkpoint as vc
from skillevaluator.validation_checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_MAGIC,
    CheckpointError,
    CheckpointStore,
    build_catalog_plan,
    external_tool_identities,
    is_reusable_success,
    plan_diff_dimension,
    plan_environment,
    plan_fingerprint,
    record_filename,
    skill_content_digest,
    skill_plan,
    validate_skill_relpath,
)

PAYLOAD = {
    "overall_passed": True,
    "overall_status": "passed",
    "incomplete_scans": [],
    "results": [{"validator": "Schema", "passed": True}],
    "severity_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
}


def _make_skill(root: Path, name: str = "sample") -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill used by checkpoint tests.\n---\n# {name}\n",
        encoding="utf-8",
    )
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.py").write_text("print('hello')\n", encoding="utf-8")
    return skill


def _base_plan(root: Path, **overrides: object) -> dict:
    plan = build_catalog_plan(
        catalog_root=root,
        content_type="skill",
        checks_raw=None,
        check_lineup=["schema"],
        policy_digest="sha256:policy",
        profile_name="external",
        policy_path=None,
        quality_config=None,
        options={"min_score": 70},
        report_formats=("json",),
        verbose=False,
    )
    plan.update(overrides)
    return plan


# ---------------------------------------------------------------------------
# Content digest
# ---------------------------------------------------------------------------


def test_content_digest_changes_with_content_and_mode(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    baseline = skill_content_digest(skill)

    (skill / "scripts" / "run.py").write_text("print('changed')\n", encoding="utf-8")
    assert skill_content_digest(skill) != baseline

    baseline2 = skill_content_digest(skill)
    new_file = skill / "notes.txt"
    new_file.write_text("new file\n", encoding="utf-8")
    assert skill_content_digest(skill) != baseline2

    baseline3 = skill_content_digest(skill)
    mode = new_file.stat().st_mode
    os.chmod(new_file, mode | stat.S_IXUSR)
    try:
        assert skill_content_digest(skill) != baseline3
    finally:
        os.chmod(new_file, mode)


def test_content_digest_ignores_scan_excluded_artifacts(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    baseline = skill_content_digest(skill)

    # Generated/artifact directories and files are outside the scanned scope.
    (skill / "evals").mkdir()
    (skill / "evals" / "evals.json").write_text('{"id": "x"}', encoding="utf-8")
    (skill / "results").mkdir()
    (skill / "results" / "run.json").write_text("{}", encoding="utf-8")
    (skill / "BENCHMARK.md").write_text("# benchmark", encoding="utf-8")
    assert skill_content_digest(skill) == baseline


def test_content_digest_records_link_targets_without_following(tmp_path: Path) -> None:
    skill = _make_skill(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("one\n", encoding="utf-8")
    link = skill / "linked.txt"
    link.symlink_to(outside)

    with_link = skill_content_digest(skill)
    # Changing the external target bytes does not change the digest: only the
    # lexical link target is part of the identity.
    outside.write_text("two-three-four\n", encoding="utf-8")
    assert skill_content_digest(skill) == with_link

    link.unlink()
    assert skill_content_digest(skill) != with_link


def test_content_digest_rejects_linked_root(tmp_path: Path) -> None:
    real = _make_skill(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(CheckpointError, match="unlinked directory"):
        skill_content_digest(alias)


# ---------------------------------------------------------------------------
# Plan identity
# ---------------------------------------------------------------------------


def test_plan_fingerprint_is_stable_for_same_plan(tmp_path: Path) -> None:
    plan = skill_plan(_base_plan(tmp_path), skill_relpath="a", content_digest="algo:aaa")
    assert plan_fingerprint(plan) == plan_fingerprint(json.loads(json.dumps(plan)))
    assert plan_fingerprint(plan).startswith(vc.PLAN_FINGERPRINT_ALGORITHM + ":")


@pytest.mark.parametrize(
    ("mutate", "dimension"),
    [
        (lambda p: p.__setitem__("content_type", "rules"), "content type"),
        (lambda p: p.__setitem__("checks", {"raw": "schema", "lineup": ["schema", "version"]}), "check set"),
        (lambda p: p.__setitem__("policy", {**p["policy"], "digest": "sha256:other"}), "policy file"),
        (
            lambda p: p.__setitem__(
                "quality_config", {"path": "/tmp/q.yaml", "sha256": "abcd"}
            ),
            "quality config",
        ),
        (lambda p: p.__setitem__("tool_version", "9.9.9"), "tool version"),
        (lambda p: p.__setitem__("options", {**p["options"], "min_score": 90}), "validation options"),
        (lambda p: p.__setitem__("report", {"formats": ["cli"], "verbose": False}), "report settings"),
        (lambda p: p.__setitem__("env", {**p["env"], "SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS": "data"}), "environment"),
        (
            lambda p: p.__setitem__("tools", {"skillspector": {"path": "/x", "stat": None}}),
            "external tool version",
        ),
    ],
)
def test_plan_fingerprint_changes_for_every_identity_dimension(
    tmp_path: Path, mutate: object, dimension: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS", raising=False)
    base = _base_plan(tmp_path)
    before = skill_plan(base, skill_relpath="a", content_digest="algo:aaa")
    after_plan = _base_plan(tmp_path)
    mutate(after_plan)
    after = skill_plan(after_plan, skill_relpath="a", content_digest="algo:aaa")
    assert plan_fingerprint(before) != plan_fingerprint(after)
    assert plan_diff_dimension(before, after) == dimension


def test_skill_content_is_its_own_fingerprint_dimension(tmp_path: Path) -> None:
    base = _base_plan(tmp_path)
    a = skill_plan(base, skill_relpath="a", content_digest="algo:aaa")
    b = skill_plan(dict(base), skill_relpath="a", content_digest="algo:bbb")
    assert plan_fingerprint(a) != plan_fingerprint(b)
    assert plan_diff_dimension(a, b) == "skill content"


def test_plan_environment_excludes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKILL_EVAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    monkeypatch.setenv("SKILLSPECTOR_MODEL_GPT", "gpt-x")
    env = plan_environment()
    assert env.get("SKILL_EVAL_LLM_PROVIDER") == "openai"
    assert env.get("SKILLSPECTOR_MODEL_GPT") == "gpt-x"
    assert "OPENAI_API_KEY" not in env
    assert all("KEY" not in name or "PROVIDER" in name or name.endswith("_PATH") for name in env)


def test_external_tool_identities_reflect_resolved_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = tmp_path / "skillspector"
    tool.write_text("#!/bin/sh\necho v1\n", encoding="utf-8")
    tool.chmod(0o755)
    monkeypatch.setenv("SKILLEVALUATOR_SKILLSPECTOR_PATH", str(tool))
    identities = external_tool_identities(["security"])
    assert identities["skillspector"] is not None
    first_stat = dict(identities["skillspector"]["stat"])

    # Replacing the binary content changes the mtime/signature.
    tool.write_text("#!/bin/sh\necho v2\n", encoding="utf-8")
    os.utime(tool, (1_000_000_000, 1_000_000_000))
    second = external_tool_identities(["security"])
    assert second["skillspector"]["stat"] != first_stat
    assert external_tool_identities(["schema"]) == {}


# ---------------------------------------------------------------------------
# File names and eligibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unsafe", ["../x", "/abs/x", "a/../../b", "a\\b", "", "."])
def test_relpath_validation_rejects_traversal(unsafe: str) -> None:
    with pytest.raises(CheckpointError):
        validate_skill_relpath(unsafe)


def test_record_filename_is_safe_and_distinct(tmp_path: Path) -> None:
    name = record_filename(tmp_path / "catalog", "team/alpha")
    assert name.startswith("skill-") and name.endswith(".json")
    assert ".." not in name and "/" not in name
    assert record_filename(tmp_path / "catalog", "team/alpha") == name
    assert record_filename(tmp_path / "catalog", "team/beta") != name
    assert record_filename(tmp_path / "other-catalog", "team/alpha") != name


def test_reusable_success_truth_table() -> None:
    assert is_reusable_success(True, PAYLOAD) is True
    assert is_reusable_success(True, None) is True  # cli-only verdict
    assert is_reusable_success(False, PAYLOAD) is False
    assert is_reusable_success(True, {**PAYLOAD, "overall_passed": False}) is False
    assert is_reusable_success(True, {**PAYLOAD, "overall_status": "incomplete"}) is False
    assert is_reusable_success(True, {**PAYLOAD, "incomplete_scans": ["skillspector"]}) is False
    assert is_reusable_success(True, {**PAYLOAD, "results": "nope"}) is False


# ---------------------------------------------------------------------------
# Store: write, classify, restore, corruption, safety
# ---------------------------------------------------------------------------


def _write_record(store: CheckpointStore, root: Path, relpath: str = "sample", **kw: object) -> str:
    plan = skill_plan(_base_plan(root), skill_relpath=relpath, content_digest="algo:aaa")
    fingerprint = plan_fingerprint(plan)
    store.write_success(
        skill_relpath=relpath,
        plan=plan,
        fingerprint=fingerprint,
        json_report_name=kw.pop("json_report_name", "skillevaluator-output-1.json"),
        json_payload=kw.pop("json_payload", PAYLOAD),
    )
    return fingerprint


def test_store_roundtrip_classify_and_restore(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    store = CheckpointStore(tmp_path / "cp", catalog)
    fingerprint = _write_record(store, catalog)
    out_dir = tmp_path / "out"

    records, warnings, _invalid = store.scan()
    assert warnings == []
    decision = store.classify("sample", skill_plan(_base_plan(catalog), skill_relpath="sample", content_digest="algo:aaa"), fingerprint, records)
    assert decision.kind == "reuse"
    restored = store.restore_json_report(decision.record, out_dir)
    assert restored == "skillevaluator-output-1.json"
    assert json.loads((out_dir / restored).read_text(encoding="utf-8")) == PAYLOAD


def test_store_requires_real_unlinked_directory(tmp_path: Path) -> None:
    target = tmp_path / "real-cp"
    target.mkdir()
    link = tmp_path / "linked-cp"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CheckpointError, match="symlink"):
        CheckpointStore(link, tmp_path / "catalog")

    (tmp_path / "a-file").write_text("x", encoding="utf-8")
    with pytest.raises(CheckpointError, match="not a directory"):
        CheckpointStore(tmp_path / "a-file", tmp_path / "catalog")


def test_store_rejects_nested_symlink_intermediate(tmp_path: Path) -> None:
    real = tmp_path / "real-dir"
    real.mkdir()
    link = tmp_path / "link-dir"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(CheckpointError, match="symlink"):
        CheckpointStore(link / "cp", tmp_path / "catalog")


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses permission bits")
def test_store_fails_closed_on_unwritable_directory(tmp_path: Path) -> None:
    readonly = tmp_path / "readonly"
    readonly.mkdir(mode=0o500)
    try:
        with pytest.raises(CheckpointError):
            CheckpointStore(readonly / "cp", tmp_path / "catalog")
    finally:
        readonly.chmod(0o700)


def test_store_creates_nested_directory_with_owner_permissions(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "cp"
    CheckpointStore(nested, tmp_path / "catalog")
    assert nested.is_dir()


def test_corrupt_and_foreign_and_incompatible_records_warn_and_stay_rejected(
    tmp_path: Path,
) -> None:
    catalog_a = tmp_path / "catalog-a"
    catalog_b = tmp_path / "catalog-b"
    cp_dir = tmp_path / "cp"
    store_a = CheckpointStore(cp_dir, catalog_a)
    _write_record(store_a, catalog_a)

    # A second catalog's records are foreign targets in the same directory.
    store_b = CheckpointStore(cp_dir, catalog_b, prepare=False)
    _write_record(store_b, catalog_b, json_report_name="other.json")
    records_a, warnings_a, _invalid_a = store_a.scan()
    assert "sample" in records_a
    assert any("different validation target" in warning for warning in warnings_a)

    # Corrupt the JSON: unusable file warning + stale (re-execute) decision.
    record_path = store_a.record_path("sample")
    record_path.write_text("{not json", encoding="utf-8")
    records, warnings, invalid = store_a.scan()
    assert "sample" not in records
    assert any("valid JSON" in warning for warning in warnings)
    plan = skill_plan(_base_plan(catalog_a), skill_relpath="sample", content_digest="algo:aaa")
    decision = store_a.classify("sample", plan, plan_fingerprint(plan), records, invalid)
    assert decision.kind == "stale"
    assert "corrupt" in decision.reason


def test_incompatible_format_version_is_rejected(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    store = CheckpointStore(tmp_path / "cp", catalog)
    _write_record(store, catalog)
    path = store.record_path("sample")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["checkpoint_version"] = CHECKPOINT_FORMAT_VERSION + 1
    path.write_text(json.dumps(record), encoding="utf-8")

    records, warnings, invalid = store.scan()
    assert any("incompatible checkpoint format version" in warning for warning in warnings)
    plan = skill_plan(_base_plan(catalog), skill_relpath="sample", content_digest="algo:aaa")
    decision = store.classify("sample", plan, plan_fingerprint(plan), records, invalid)
    assert decision.kind == "stale"
    assert "incompatible" in decision.reason


def test_tampered_plan_fingerprint_is_rejected(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    store = CheckpointStore(tmp_path / "cp", catalog)
    fingerprint = _write_record(store, catalog)
    path = store.record_path("sample")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["plan"]["options"] = {"min_score": 99}
    path.write_text(json.dumps(record), encoding="utf-8")

    records, warnings, invalid = store.scan()
    assert any("fingerprint" in warning for warning in warnings)
    assert "fingerprint" in invalid.get("sample", "")
    plan = skill_plan(_base_plan(catalog), skill_relpath="sample", content_digest="algo:aaa")
    decision = store.classify("sample", plan, fingerprint, records, invalid)
    assert decision.kind == "stale"


def test_write_success_rejects_incomplete_payload_and_unsafe_names(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    store = CheckpointStore(tmp_path / "cp", catalog)
    plan = skill_plan(_base_plan(catalog), skill_relpath="sample", content_digest="algo:aaa")
    with pytest.raises(CheckpointError, match="incomplete"):
        store.write_success(
            skill_relpath="sample",
            plan=plan,
            fingerprint=plan_fingerprint(plan),
            json_report_name="r.json",
            json_payload={**PAYLOAD, "incomplete_scans": ["semgrep"]},
        )
    with pytest.raises(CheckpointError, match="unsafe"):
        store.write_success(
            skill_relpath="sample",
            plan=plan,
            fingerprint=plan_fingerprint(plan),
            json_report_name="../escape.json",
            json_payload=PAYLOAD,
        )
    with pytest.raises(CheckpointError, match="standard JSON"):
        store.write_success(
            skill_relpath="sample",
            plan=plan,
            fingerprint=plan_fingerprint(plan),
            json_report_name="r.sarif.json",
            json_payload=PAYLOAD,
        )


def test_writes_are_independent_and_survive_concurrency(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    store = CheckpointStore(tmp_path / "cp", catalog)

    def write_one(index: int) -> None:
        relpath = f"skill-{index}"
        plan = skill_plan(_base_plan(catalog), skill_relpath=relpath, content_digest=f"algo:{index}")
        store.write_success(
            skill_relpath=relpath,
            plan=plan,
            fingerprint=plan_fingerprint(plan),
            json_report_name=None,
            json_payload=None,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_one, range(24)))

    records, warnings, _invalid = store.scan()
    assert warnings == []
    assert set(records) == {f"skill-{i}" for i in range(24)}


def test_magic_guards_foreign_files(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    store = CheckpointStore(tmp_path / "cp", catalog)
    _write_record(store, catalog)
    path = store.record_path("sample")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["magic"] = "something-else"
    path.write_text(json.dumps(record), encoding="utf-8")
    records, warnings, invalid = store.scan()
    assert any("unknown checkpoint format" in warning for warning in warnings)
    plan = skill_plan(_base_plan(catalog), skill_relpath="sample", content_digest="algo:aaa")
    assert store.classify("sample", plan, plan_fingerprint(plan), records, invalid).kind == "stale"


def test_checkpoint_constants_are_stable() -> None:
    assert CHECKPOINT_MAGIC == "skillevaluator-validate-checkpoint"
    assert CHECKPOINT_FORMAT_VERSION == 1
