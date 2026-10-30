# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable checkpoints for catalog (whole-directory) validation.

When ``validate`` scans a directory of skills, each discovered skill is an
independent job. A long run interrupted by a closed terminal, sleeping
machine, or cancelled CI job must not rescan skills that already completed
and whose inputs have not changed. ``--checkpoint-dir`` therefore persists,
per skill and atomically:

* the exact validation *plan* the run used (content digest, content type,
  check set, policy, quality config, tool/external-scanner identity,
  result-affecting options, report settings, and relevant environment);
* the skill's successful result (the gate verdict and the JSON report
  payload the catalog summary is assembled from).

On the next invocation with the same command, a record is reused only when
its plan fingerprint matches the current plan in full. Any mismatch, a
corrupt/foreign/version-incompatible record, a failure, or an incomplete
scan invalidates the record and the skill is re-executed. Records for
deleted skills are left on disk as history and never counted for the
current run.

The module never trusts the filesystem: checkpoint directories must be
real, non-symlinked, writable directories; record names are derived from
hashes (skill paths never reach the filesystem literally); all writes go
through the same link-rejecting, atomic report writer as the reporters.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillevaluator import __version__
from skillevaluator.constants import SCAN_EXCLUDED_DIRS, SCAN_EXCLUDED_FILES
from skillevaluator.reporting.base import UnsafeReportPathError, _write_report_atomically
from skillevaluator.utils.path_security import canonicalize_trusted_root_alias, matches_filesystem_name
from skillevaluator.utils.tier2_paths import is_link_or_reparse
from skillevaluator.utils.tool_runner import resolve_tool_path

#: Magic prefix every checkpoint record starts with.
CHECKPOINT_MAGIC = "skillevaluator-validate-checkpoint"

#: Record format this package writes and accepts. A different major format
#: (tampered, or produced by an incompatible future/older release) is
#: rejected with a warning rather than trusted.
CHECKPOINT_FORMAT_VERSION = 1

RECORD_FILE_PREFIX = "skill-"
RECORD_FILE_SUFFIX = ".json"

PLAN_FINGERPRINT_ALGORITHM = "skillevaluator-validate-plan/1"
CONTENT_DIGEST_ALGORITHM = "skillevaluator-skill-content/1"

# A guard against decompression-/disk-fill style garbage in a directory an
# operator points at; real records stay far below this.
MAX_RECORD_BYTES = 64 * 1024 * 1024

# Result-affecting environment that is safe to record (no credentials).
# Provider *selection* changes static/dedup results; provider *keys* do not
# belong here and are deliberately excluded.
_PLAN_ENV_VARS = (
    "SKILLEVALUATOR_PROFILE",
    "SKILLEVALUATOR_PREVIOUS_VERSION",
    "SKILLEVALUATOR_SCHEMA_ALLOWED_DIRS",
    "SKILLEVALUATOR_SUBMITTER",
    "GITHUB_ACTOR",
    "USER",
    "LOGNAME",
    "USERNAME",
    "SKILLEVALUATOR_BANDIT_PATH",
    "SKILLEVALUATOR_SEMGREP_PATH",
    "SKILLEVALUATOR_SKILLSPECTOR_PATH",
    "SKILLSPECTOR_PROVIDER",
    "SKILLSPECTOR_MODEL",
    "SKILLSPECTOR_REASONING_EFFORT",
    "SKILL_EVAL_LLM_PROVIDER",
    "SKILL_EVAL_LLM_MODEL",
    "SKILL_EVAL_LLM_BASE_URL",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "LLM_JUDGE_FALLBACK_MODELS",
    "SKILL_EVAL_BEHAVIOR_CHECK_BUDGET",
)
_PLAN_ENV_PREFIXES = ("SKILLSPECTOR_MODEL_",)

# check name -> external scanners whose results that check can depend on.
# (tool command, prefer interpreter sibling, path-override env var).
_CHECK_TOOLS: dict[str, tuple[tuple[str, bool, str | None], ...]] = {
    "security": (("skillspector", True, "SKILLEVALUATOR_SKILLSPECTOR_PATH"),),
    "code-integrity": (
        ("bandit", True, "SKILLEVALUATOR_BANDIT_PATH"),
        ("semgrep", True, "SKILLEVALUATOR_SEMGREP_PATH"),
        ("gitleaks", False, None),
    ),
    "dependency": (
        ("pip-audit", True, None),
        ("safety", False, None),
    ),
}


class CheckpointError(Exception):
    """Raised when checkpoint storage cannot be used safely."""


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_hash(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "sha256": None}
    resolved = path.expanduser()
    try:
        return {"path": str(resolved), "sha256": _hash_file(resolved)}
    except OSError as exc:
        raise CheckpointError(f"Cannot hash {resolved}: {exc.strerror or exc}") from exc


def plan_environment() -> dict[str, str]:
    """Return the sorted, secret-free environment slice a plan depends on."""
    selected = {name: os.environ[name] for name in _PLAN_ENV_VARS if os.environ.get(name, "").strip()}
    selected.update(
        {name: value for name, value in os.environ.items() if name.startswith(_PLAN_ENV_PREFIXES) and value.strip()}
    )
    return dict(sorted(selected.items()))


def _tool_identity(command: str, prefer_sibling: bool, override_env: str | None) -> dict[str, Any] | None:
    """Identify an external scanner cheaply, without spawning it.

    The resolved absolute executable plus its inode/size/mtime signature
    detects a swapped or upgraded binary; missing tools record ``None`` so
    installing one also changes the plan.
    """
    override = os.environ.get(override_env, "").strip() if override_env else ""
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            # The tool wrapper itself fails closed on a relative override;
            # mirror that instead of silently scanning a different binary.
            return {"path": None, "override": override, "error": "override is not absolute"}
        resolved = candidate
    else:
        located = resolve_tool_path(command, prefer_interpreter_sibling=prefer_sibling)
        if located is None:
            return None
        resolved = Path(located)
    try:
        metadata = resolved.lstat()
    except OSError:
        return {"path": str(resolved), "stat": None}
    if not stat.S_ISREG(metadata.st_mode):
        return {"path": str(resolved), "stat": None}
    return {
        "path": str(resolved),
        "override": override or None,
        "stat": {
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "inode": (metadata.st_dev, metadata.st_ino),
        },
    }


def external_tool_identities(check_lineup: list[str]) -> dict[str, dict[str, Any] | None]:
    identities: dict[str, dict[str, Any] | None] = {}
    for check in check_lineup:
        for command, prefer_sibling, override_env in _CHECK_TOOLS.get(check, ()):
            if command not in identities:
                identities[command] = _tool_identity(command, prefer_sibling, override_env)
    return dict(sorted(identities.items()))


def skill_content_digest(skill_dir: Path) -> str:
    """Hash the scanned content of one skill directory.

    Mirrors what the Tier 1 walk can observe: scan-excluded artifact
    directories (``evals/``, ``results/``, ``.git/``, ...) and generated
    files are pruned, linked directories are recorded as link markers and
    never followed, and each regular file contributes its POSIX-relative
    path, permission bits, and bytes. Symlinked files contribute the link
    target rather than external bytes. Any unreadable entry fails closed so
    a content change can never be silently missed.
    """
    directory = skill_dir
    try:
        root_metadata = directory.lstat()
    except OSError as exc:
        raise CheckpointError(f"Cannot stat skill directory {directory}: {exc.strerror or exc}") from exc
    if is_link_or_reparse(directory) or not stat.S_ISDIR(root_metadata.st_mode):
        raise CheckpointError(f"Skill content root must be an unlinked directory: {directory}")

    entries: list[dict[str, Any]] = []

    def on_walk_error(error: OSError) -> None:
        raise CheckpointError(f"Cannot walk skill content under {directory}: {error}")

    for dirpath, dirnames, filenames in os.walk(directory, onerror=on_walk_error, followlinks=False):
        base = Path(dirpath)
        retained_dirs: list[str] = []
        for name in dirnames:
            child = base / name
            if matches_filesystem_name(child, SCAN_EXCLUDED_DIRS):
                continue
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise CheckpointError(f"Cannot stat {child}: {exc.strerror or exc}") from exc
            relpath = child.relative_to(directory).as_posix()
            if stat.S_ISLNK(metadata.st_mode) or is_link_or_reparse(child):
                try:
                    entries.append({"kind": "link-dir", "path": relpath, "target": os.fspath(child.readlink())})
                except OSError as exc:
                    raise CheckpointError(f"Cannot read link {child}: {exc.strerror or exc}") from exc
                continue
            retained_dirs.append(name)
        dirnames[:] = retained_dirs

        for name in sorted(filenames):
            child = base / name
            if matches_filesystem_name(child, SCAN_EXCLUDED_FILES):
                continue
            relpath = child.relative_to(directory).as_posix()
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise CheckpointError(f"Cannot stat {child}: {exc.strerror or exc}") from exc
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    entries.append({"kind": "link-file", "path": relpath, "target": os.fspath(child.readlink())})
                except OSError as exc:
                    raise CheckpointError(f"Cannot read link {child}: {exc.strerror or exc}") from exc
                continue
            if not stat.S_ISREG(metadata.st_mode):
                entries.append({"kind": "special", "path": relpath, "mode": mode})
                continue
            try:
                content_hash = _hash_file(child)
            except OSError as exc:
                raise CheckpointError(f"Cannot read {child}: {exc.strerror or exc}") from exc
            entries.append({"kind": "file", "path": relpath, "mode": mode, "sha256": content_hash})

    entries.sort(key=lambda entry: (entry["kind"], entry["path"]))
    canonical = _canonical_json({"algorithm": CONTENT_DIGEST_ALGORITHM, "root": str(directory), "entries": entries})
    return f"{CONTENT_DIGEST_ALGORITHM}:{_sha256_bytes(canonical)}"


def path_content_digest(path: Path) -> str:
    """Digest an extra mounted path (``--include-skills``), file or directory."""
    if path.is_dir() and not is_link_or_reparse(path):
        return skill_content_digest(path)
    metadata = path.lstat()
    payload = {
        "algorithm": CONTENT_DIGEST_ALGORITHM,
        "path": str(path),
        "mode": stat.S_IMODE(metadata.st_mode),
        "target": os.fspath(path.readlink()) if stat.S_ISLNK(metadata.st_mode) else None,
        "sha256": _hash_file(path) if stat.S_ISREG(metadata.st_mode) else None,
    }
    return f"{CONTENT_DIGEST_ALGORITHM}:{_sha256_bytes(_canonical_json(payload))}"


def build_catalog_plan(
    *,
    catalog_root: Path,
    content_type: str,
    checks_raw: str | None,
    check_lineup: list[str],
    policy_digest: str,
    profile_name: str | None,
    policy_path: Path | None,
    quality_config: Path | None,
    options: dict[str, Any],
    report_formats: tuple[str, ...] | list[str],
    verbose: bool,
    include_skills: tuple[Path, ...] | list[Path] = (),
) -> dict[str, Any]:
    """Build the run-global (skill-independent) portion of a catalog plan.

    Per-skill content is added later with :func:`skill_plan`; the resulting
    full plan's :func:`plan_fingerprint` must match for a result to be
    reused.
    """
    include = []
    for extra in include_skills:
        resolved = extra.expanduser().resolve()
        try:
            include.append({"path": str(resolved), "content": path_content_digest(resolved)})
        except OSError as exc:
            raise CheckpointError(f"Cannot fingerprint --include-skills path {resolved}: {exc}") from exc
    include.sort(key=lambda item: item["path"])

    return {
        "algorithm": PLAN_FINGERPRINT_ALGORITHM,
        "catalog_root": str(catalog_root),
        "tool_version": __version__,
        "python": platform.python_version(),
        "platform": sys_platform(),
        "content_type": content_type,
        "checks": {"raw": checks_raw, "lineup": list(check_lineup)},
        "policy": {
            "profile": profile_name,
            "digest": policy_digest,
            "file": _optional_file_hash(policy_path),
        },
        "quality_config": _optional_file_hash(quality_config),
        "options": _canonical_options(options),
        "report": {"formats": list(report_formats), "verbose": bool(verbose)},
        "env": plan_environment(),
        "tools": external_tool_identities(check_lineup),
        "include_skills": include,
    }


def sys_platform() -> str:
    """Indirection so tests can exercise every plan dimension cheaply."""
    import sys

    return sys.platform


def _canonical_options(options: dict[str, Any]) -> dict[str, Any]:
    return dict(sorted(options.items()))


def skill_plan(base_plan: dict[str, Any], *, skill_relpath: str, content_digest: str) -> dict[str, Any]:
    """Return the full, per-skill plan (a fresh JSON-serializable dict)."""
    plan = dict(base_plan)
    plan["skill"] = {"path": validate_skill_relpath(skill_relpath), "content_digest": content_digest}
    return plan


def plan_fingerprint(plan: dict[str, Any]) -> str:
    return f"{PLAN_FINGERPRINT_ALGORITHM}:{_sha256_bytes(_canonical_json(plan))}"


def validate_skill_relpath(relpath: str) -> str:
    """Reject anything that is not a clean catalog-relative POSIX path."""
    if not isinstance(relpath, str) or not relpath:
        raise CheckpointError("empty skill path in checkpoint record")
    if relpath.startswith("/") or "\\" in relpath or ".." in relpath.split("/") or relpath in {".", ""}:
        raise CheckpointError(f"unsafe skill path in checkpoint record: {relpath!r}")
    return relpath


# ---------------------------------------------------------------------------
# Result eligibility
# ---------------------------------------------------------------------------


def is_reusable_success(passed: bool, payload: dict[str, Any] | None) -> bool:
    """A result may be cached/reused only when it passed and ran completely.

    Failures, incomplete external scans (missing skillspector/semgrep/...),
    and malformed payloads never become reusable records, so an interrupted
    or partial run cannot masquerade as a clean one.
    """
    if not passed:
        return False
    if payload is None:
        # cli-only (no JSON report) catalog jobs still carry a gate verdict.
        return True
    if not isinstance(payload, dict):
        return False
    if payload.get("overall_passed") is not True:
        return False
    if payload.get("overall_status") == "incomplete":
        return False
    incomplete = payload.get("incomplete_scans")
    if not isinstance(incomplete, list) or incomplete:
        return False
    return isinstance(payload.get("results"), list)


# ---------------------------------------------------------------------------
# Record storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointDecision:
    """What the catalog orchestrator should do with one skill."""

    kind: str  # "new" | "reuse" | "stale"
    record: dict[str, Any] | None = None
    reason: str = ""


def record_filename(catalog_root: Path, skill_relpath: str) -> str:
    """Map a (catalog, skill) pair to one safe, collision-free file name."""
    relpath = validate_skill_relpath(skill_relpath)
    material = f"{catalog_root}\0{relpath}".encode("utf-8")
    return f"{RECORD_FILE_PREFIX}{_sha256_bytes(material)[:32]}{RECORD_FILE_SUFFIX}"


def _is_record_file(path: Path) -> bool:
    return path.name.startswith(RECORD_FILE_PREFIX) and path.name.endswith(RECORD_FILE_SUFFIX)


def _claimed_skill(path: Path) -> str | None:
    """Best-effort extraction of a structurally invalid record's skill key."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    target = raw.get("target")
    if not isinstance(target, dict):
        return None
    skill = target.get("skill")
    if not isinstance(skill, str):
        return None
    try:
        return validate_skill_relpath(skill)
    except CheckpointError:
        return None


class CheckpointStore:
    """Owns checkpoint directory setup, record persistence, and lookup."""

    def __init__(self, directory: Path, catalog_root: Path, *, prepare: bool = True) -> None:
        try:
            leaf = directory.expanduser()
            if not leaf.is_absolute():
                leaf = Path.cwd() / leaf
            # Honor root-owned POSIX aliases such as /tmp -> /private/tmp so
            # the strict no-symlink component walk still accepts them, while
            # later components and the leaf itself must be real directories.
            leaf = canonicalize_trusted_root_alias(leaf)
            self.directory = leaf
            self.catalog_root = catalog_root.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointError(f"Cannot resolve checkpoint directory {directory}: {exc}") from exc

        self._ensure_real_directory()
        if prepare:
            self.probe_writable()

    # -- directory safety -------------------------------------------------

    def _ensure_real_directory(self) -> None:
        try:
            metadata = self.directory.lstat()
        except FileNotFoundError:
            self._create_directory()
            return
        except OSError as exc:
            raise CheckpointError(f"Cannot inspect checkpoint directory {self.directory}: {exc.strerror or exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or is_link_or_reparse(self.directory):
            raise CheckpointError(f"Checkpoint directory is a symlink or reparse point: {self.directory}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise CheckpointError(f"Checkpoint path is not a directory: {self.directory}")

    def _create_directory(self) -> None:
        # Create component by component, refusing a symlinked intermediate.
        current = Path(self.directory.anchor)
        for component in self.directory.parts[1:]:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                try:
                    os.mkdir(current, 0o700)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise CheckpointError(
                        f"Cannot create checkpoint directory {self.directory}: {exc.strerror or exc}"
                    ) from exc
                try:
                    metadata = current.lstat()
                except OSError as exc:
                    raise CheckpointError(f"Cannot inspect {current}: {exc.strerror or exc}") from exc
            except OSError as exc:
                raise CheckpointError(f"Cannot inspect {current}: {exc.strerror or exc}") from exc
            if stat.S_ISLNK(metadata.st_mode) or is_link_or_reparse(current):
                raise CheckpointError(f"Checkpoint path traverses a symlink or reparse point: {current}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise CheckpointError(f"Checkpoint path component is not a directory: {current}")

    def probe_writable(self) -> None:
        """Fail before scanning when checkpoints could not be persisted."""
        probe = self.directory / f".write-probe-{secrets.token_hex(8)}"
        try:
            _write_report_atomically(probe, b"ok")
            probe.unlink()
        except UnsafeReportPathError as exc:
            raise CheckpointError(str(exc)) from exc
        except OSError as exc:
            raise CheckpointError(
                f"Checkpoint directory is not writable {self.directory}: {exc.strerror or exc}"
            ) from exc

    # -- paths -------------------------------------------------------------

    def record_path(self, skill_relpath: str) -> Path:
        return self.directory / record_filename(self.catalog_root, validate_skill_relpath(skill_relpath))

    # -- write -------------------------------------------------------------

    def write_success(
        self,
        *,
        skill_relpath: str,
        plan: dict[str, Any],
        fingerprint: str,
        json_report_name: str | None,
        json_payload: dict[str, Any] | None,
    ) -> None:
        """Atomically persist one complete, successful per-skill result."""
        relpath = validate_skill_relpath(skill_relpath)
        if not is_reusable_success(True, json_payload):
            raise CheckpointError(f"refusing to checkpoint an incomplete result for {relpath}")
        if json_report_name is not None:
            self._validate_json_report_name(json_report_name)
        if plan_fingerprint(plan) != fingerprint:
            raise CheckpointError(f"plan fingerprint mismatch while checkpointing {relpath}")
        record = {
            "magic": CHECKPOINT_MAGIC,
            "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
            "tool_version": plan.get("tool_version"),
            "target": {"catalog_root": str(self.catalog_root), "skill": relpath},
            "plan": plan,
            "plan_fingerprint": fingerprint,
            "status": "success",
            "passed": True,
            "json_report": (
                None
                if json_report_name is None
                else {"name": json_report_name, "payload": json_payload}
            ),
            "recorded_at": datetime.now(tz=UTC).isoformat(),
        }
        payload = json.dumps(record, indent=2, allow_nan=False, default=str).encode("utf-8")
        try:
            _write_report_atomically(self.record_path(relpath), payload)
        except UnsafeReportPathError as exc:
            raise CheckpointError(str(exc)) from exc

    @staticmethod
    def _validate_json_report_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise CheckpointError("empty JSON report name in checkpoint record")
        if Path(name).name != name or name in {".", ".."}:
            raise CheckpointError(f"unsafe JSON report name in checkpoint record: {name!r}")
        if not name.endswith(".json") or name.endswith(".sarif.json"):
            raise CheckpointError(f"checkpoint did not bind the standard JSON report: {name!r}")

    # -- read --------------------------------------------------------------

    def scan(self) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, str]]:
        """Load every parseable record belonging to this catalog root.

        Returns ``(records_by_skill, warnings, invalid)``. Corrupt files and
        records from other targets are reported as warnings and left on disk
        as history; they never enter the reusable map. Structurally invalid
        records whose target skill can still be identified are keyed in
        ``invalid`` so :meth:`classify` reports the precise reason.
        """
        records: dict[str, dict[str, Any]] = {}
        invalid: dict[str, str] = {}
        warnings: list[str] = []
        try:
            children = sorted(self.directory.iterdir())
        except OSError as exc:
            raise CheckpointError(f"Cannot read checkpoint directory {self.directory}: {exc.strerror or exc}") from exc
        for child in children:
            if not _is_record_file(child):
                continue
            try:
                metadata = child.lstat()
            except OSError:
                warnings.append(f"Ignoring unreadable checkpoint file {child.name}")
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                warnings.append(f"Ignoring non-regular checkpoint file {child.name}")
                continue
            record, reason = self._read_record(child)
            if record is None:
                warnings.append(f"Ignoring unusable checkpoint file {child.name}: {reason}")
                # When the JSON itself parsed but a structural check failed,
                # keep the reason keyed by its claimed target skill so the
                # catalog can report *why* that skill is being re-executed.
                candidate = _claimed_skill(child)
                if candidate is not None:
                    invalid[candidate] = reason
                continue
            target = record.get("target")
            if not isinstance(target, dict) or target.get("catalog_root") != str(self.catalog_root):
                warnings.append(
                    f"Ignoring checkpoint file {child.name} from a different validation target; re-executing"
                )
                continue
            relpath = target.get("skill")
            if not isinstance(relpath, str):
                warnings.append(f"Ignoring checkpoint file {child.name}: missing target skill")
                continue
            try:
                validate_skill_relpath(relpath)
            except CheckpointError:
                warnings.append(f"Ignoring checkpoint file {child.name}: unsafe target skill path")
                continue
            records[relpath] = record
        return records, warnings, invalid

    def _read_record(self, path: Path) -> tuple[dict[str, Any] | None, str]:
        try:
            size = path.stat().st_size
        except OSError as exc:
            return None, f"cannot stat ({exc.strerror or exc})"
        if size > MAX_RECORD_BYTES:
            return None, "record is larger than the supported size"
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return None, f"cannot read ({exc.strerror or exc})"
        try:
            record = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "content is not valid JSON"
        if not isinstance(record, dict):
            return None, "record is not a JSON object"
        reason = self.record_invalid_reason(record)
        if reason:
            return None, reason
        return record, ""

    @staticmethod
    def record_invalid_reason(record: dict[str, Any]) -> str:
        """Structural validation shared by directory scans and lookups."""
        if record.get("magic") != CHECKPOINT_MAGIC:
            return "unknown checkpoint format or foreign file"
        version = record.get("checkpoint_version")
        if not isinstance(version, int) or isinstance(version, bool):
            return "missing checkpoint format version"
        if version != CHECKPOINT_FORMAT_VERSION:
            return f"incompatible checkpoint format version {version}"
        target = record.get("target")
        if not isinstance(target, dict) or not isinstance(target.get("catalog_root"), str):
            return "missing validation target"
        if not isinstance(target.get("skill"), str):
            return "missing target skill"
        plan = record.get("plan")
        fingerprint = record.get("plan_fingerprint")
        if not isinstance(plan, dict) or not isinstance(fingerprint, str):
            return "missing validation plan"
        if plan_fingerprint(plan) != fingerprint:
            return "plan fingerprint does not match the stored plan"
        if record.get("status") != "success" or record.get("passed") is not True:
            return "record is not a completed success"
        json_report = record.get("json_report")
        if json_report is not None:
            if not isinstance(json_report, dict):
                return "malformed stored JSON report"
            name = json_report.get("name")
            payload = json_report.get("payload")
            if not isinstance(name, str) or not isinstance(payload, dict):
                return "malformed stored JSON report"
            if Path(name).name != name or not name.endswith(".json") or name.endswith(".sarif.json"):
                return "unsafe stored JSON report name"
        recorded_at = record.get("recorded_at")
        if not isinstance(recorded_at, str) or not recorded_at:
            return "missing record timestamp"
        return ""

    def classify(
        self,
        skill_relpath: str,
        current_plan: dict[str, Any],
        current_fingerprint: str,
        records: dict[str, dict[str, Any]],
        invalid: dict[str, str] | None = None,
    ) -> CheckpointDecision:
        """Decide reuse vs. re-execution for one skill of the current run."""
        relpath = validate_skill_relpath(skill_relpath)
        record = records.get(relpath)
        if record is None:
            if invalid and relpath in invalid:
                return CheckpointDecision("stale", reason=invalid[relpath])
            # A record-shaped file at this skill's expected name that failed
            # validation is a corrupt/foreign record: re-execute rather than
            # treating the skill as brand new (which would hide the warning).
            try:
                metadata = self.record_path(relpath).lstat()
            except FileNotFoundError:
                return CheckpointDecision("new")
            except OSError:
                return CheckpointDecision("stale", reason="checkpoint record cannot be inspected")
            if stat.S_ISLNK(metadata.st_mode) or is_link_or_reparse(self.record_path(relpath)):
                return CheckpointDecision("stale", reason="checkpoint record path is a symlink")
            if stat.S_ISREG(metadata.st_mode):
                return CheckpointDecision("stale", reason="checkpoint record is corrupt or unreadable")
            return CheckpointDecision("stale", reason="checkpoint record is not a regular file")
        reason = self.record_invalid_reason(record)
        if reason:
            return CheckpointDecision("stale", reason=reason)
        target = record["target"]
        if target.get("catalog_root") != str(self.catalog_root) or target.get("skill") != relpath:
            return CheckpointDecision("stale", reason="record belongs to a different validation target")
        stored_fingerprint = record.get("plan_fingerprint")
        if stored_fingerprint != current_fingerprint:
            dimension = plan_diff_dimension(record.get("plan"), current_plan)
            return CheckpointDecision(
                "stale",
                reason=f"validation plan changed ({dimension})" if dimension else "validation plan changed",
            )
        json_report = record.get("json_report")
        payload = json_report.get("payload") if isinstance(json_report, dict) else None
        if not is_reusable_success(True, payload):
            return CheckpointDecision("stale", reason="stored result failed or reported incomplete scans")
        return CheckpointDecision("reuse", record=record)

    # -- restore -----------------------------------------------------------

    def restore_json_report(self, record: dict[str, Any], skill_output_dir: Path) -> str | None:
        """Recreate the per-skill JSON report a fresh run would have written.

        Returns the report basename (already recorded in the catalog summary
        via the normal binding path), or ``None`` for cli-only records.
        """
        json_report = record.get("json_report")
        if not isinstance(json_report, dict):
            return None
        name = json_report["name"]
        self._validate_json_report_name(name)
        payload = json_report["payload"]
        destination = skill_output_dir / name
        body = json.dumps(payload, indent=2, allow_nan=False, default=str).encode("utf-8")
        try:
            _write_report_atomically(destination, body)
        except UnsafeReportPathError as exc:
            raise CheckpointError(str(exc)) from exc
        except OSError as exc:
            raise CheckpointError(f"Cannot restore JSON report for {record.get('target')}: {exc}") from exc
        return name


def plan_diff_dimension(stored: Any, current: Any) -> str:
    """Name the first plan dimension that differs, for a resume warning."""
    labels = {
        "skill": "skill content",
        "content_type": "content type",
        "checks": "check set",
        "policy": "policy file",
        "quality_config": "quality config",
        "tool_version": "tool version",
        "python": "Python version",
        "platform": "platform",
        "options": "validation options",
        "report": "report settings",
        "env": "environment",
        "tools": "external tool version",
        "include_skills": "included skills",
        "catalog_root": "validation target",
    }
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return ""
    for key in (*labels.keys(),):
        if stored.get(key) != current.get(key):
            return labels[key]
    return "validation plan"
