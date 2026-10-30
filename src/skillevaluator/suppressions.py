# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reviewable finding exemptions.

Teams that have already triaged a specific check can record the accepted
findings in a reviewable ignore file instead of filtering report output:

* ``--ignore-file PATH`` loads an explicit exemption document.
* Without it, ``skillevaluator-ignore.yaml`` is discovered at the root of the
  target skill (``--strict`` disables that discovery).

Every rule names a normalized ``CATEGORY.check_name`` and *at least one*
scoped path glob; a blanket wildcard (``*`` / ``**``) cannot disable a whole
check. Rules may carry a human ``reason`` and an optional ``expires`` date.

Matched, unexpired findings are removed from the gate but stay visible in
every report (CLI, JSON, HTML, Markdown, SARIF) together with the rule that
suppressed them and why. Expired rules, rules that never matched a finding,
and rules that tried to waive a non-exemptable finding (incomplete scans,
missing evidence, secret findings) produce visible warnings.

All path handling is purely lexical and skill-relative: absolute and
traversal paths are rejected, globs are compiled against safe POSIX-style
relative paths, and platform case conventions are honored.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from skillevaluator.models.result import ValidationResult

#: File auto-discovered in a target skill root when no ``--ignore-file`` is given.
IGNORE_FILE_NAME = "skillevaluator-ignore.yaml"

#: Metadata key carrying the run-wide :class:`SuppressionLedger` payload.
SUPPRESSION_LEDGER_KEY = "suppression_ledger"
#: Metadata key carrying finding records removed from a result by an exemption.
SUPPRESSED_FINDINGS_KEY = "suppressed_findings"

_TOP_LEVEL_KEYS = frozenset({"version", "ignores"})
_RULE_KEYS = frozenset({"id", "check", "paths", "reason", "expires"})
_RULE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_CHECK_CATEGORY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_CHECK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_WILDCARD_META = frozenset("*?[]")
_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")

# Categories that can never be exempted: hard-coded secrets stay blocking.
_NON_EXEMPTABLE_CATEGORIES = frozenset({"SECRET"})


class IgnoreFileError(ValueError):
    """Raised when an ignore document cannot be parsed or safely applied."""


@dataclass(frozen=True)
class IgnoreRule:
    """One validated exemption rule.

    Attributes:
        rule_id: Stable identifier used in reports (explicit ``id`` or ``rule-N``).
        index: Zero-based position in the source document.
        category: Normalized category (upper case), e.g. ``SCHEMA``.
        check_name: Specific check identifier, e.g. ``name_consistency``.
        paths: Safe, scoped path globs relative to the skill root.
        reason: Human-readable justification for the exemption.
        expires: Optional date after which the rule stops suppressing.
        source: Ignore document the rule came from.
    """

    rule_id: str
    index: int
    category: str
    check_name: str
    paths: tuple[str, ...]
    reason: str | None
    expires: date | None
    source: Path

    @property
    def check_id(self) -> str:
        """Canonical ``CATEGORY.check_name`` identifier."""
        return f"{self.category}.{self.check_name}"

    def is_expired(self, today: date) -> bool:
        return self.expires is not None and today > self.expires

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule_id,
            "check": self.check_id,
            "paths": list(self.paths),
            "reason": self.reason,
            "expires": self.expires.isoformat() if self.expires is not None else None,
        }


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader rejecting repeated mapping keys (last-one-wins is unsafe)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None)
    if mark is None or problem is None:
        return str(exc)
    return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"


def _normalize_check(value: object, *, source: str, where: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value.strip():
        raise IgnoreFileError(f"{source}: {where}.check must be a non-empty 'CATEGORY.check_name' string")
    text = value.strip()
    if text.count(".") < 1:
        raise IgnoreFileError(
            f"{source}: {where}.check must be 'CATEGORY.check_name' (got {value!r}); a category and a "
            "specific check name are both required"
        )
    category, check_name = text.split(".", 1)
    category = category.strip()
    check_name = check_name.strip()
    if not _CHECK_CATEGORY_PATTERN.fullmatch(category):
        raise IgnoreFileError(f"{source}: {where}.check has an invalid category: {category!r}")
    if not check_name or check_name in {"*", "**"} or not _CHECK_NAME_PATTERN.fullmatch(check_name):
        raise IgnoreFileError(f"{source}: {where}.check must name a specific check, not a wildcard (got {value!r})")
    return category.upper(), check_name


def _validate_glob(value: object, *, source: str, where: str) -> str:
    """Validate one path glob and return its canonical POSIX-style form."""
    if not isinstance(value, str) or not value.strip():
        raise IgnoreFileError(f"{source}: {where}.paths entries must be non-empty path glob strings")
    pattern = value.strip()
    if "\x00" in pattern:
        raise IgnoreFileError(f"{source}: {where}.paths contains a NUL byte: {value!r}")
    # Backslashes are filename bytes on POSIX; never treat them as separators.
    if os.name != "nt" and "\\" in pattern:
        raise IgnoreFileError(f"{source}: {where}.paths must use '/' separators (backslash is not allowed): {value!r}")
    pattern = pattern.replace("\\", "/")
    if pattern.startswith("/") or _DRIVE_PATTERN.match(pattern):
        raise IgnoreFileError(f"{source}: {where}.paths must be relative to the skill root: {value!r}")
    segments = pattern.split("/")
    if any(segment == "" for segment in segments):
        raise IgnoreFileError(
            f"{source}: {where}.paths must not contain empty path segments (leading/trailing/double '/'): {value!r}"
        )
    for segment in segments:
        if segment in {".", ".."}:
            raise IgnoreFileError(
                f"{source}: {where}.paths must stay inside the skill root ('.'/'..' are not allowed): {value!r}"
            )
        if "**" in segment and segment != "**":
            raise IgnoreFileError(f"{source}: {where}.paths may use '**' only as a full path segment: {value!r}")
        if segment != "**":
            # Reject unbalanced character classes instead of letting fnmatch
            # silently treat '[' as a literal, which never matches intent.
            depth = 0
            for char in segment:
                if char == "[":
                    depth += 1
                elif char == "]" and depth:
                    depth -= 1
            if depth:
                raise IgnoreFileError(f"{source}: {where}.paths has an unbalanced '[' glob class: {value!r}")
            try:
                re.compile(fnmatch.translate(segment))
            except re.error as exc:
                raise IgnoreFileError(f"{source}: {where}.paths is not a valid glob: {value!r} ({exc})") from exc
    # Every glob must actually scope the rule: at least one literal character
    # somewhere. '*', '**', '**/*', '???' alone would disable the whole check.
    literal = "".join(char for char in pattern if char not in _WILDCARD_META and char != "/")
    if not literal:
        raise IgnoreFileError(
            f"{source}: {where}.paths must scope the rule with a concrete path or name; an all-wildcard "
            f"glob cannot suppress the whole check: {value!r}"
        )
    return pattern


def _coerce_expires(value: object, *, source: str, where: str) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise IgnoreFileError(f"{source}: {where}.expires must be a YYYY-MM-DD date string")
    text = value.strip()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise IgnoreFileError(f"{source}: {where}.expires must be a YYYY-MM-DD date (got {value!r})") from exc


def _rule_from_entry(entry: object, *, index: int, source: Path, used_ids: set[str]) -> IgnoreRule:
    where = f"ignores[{index}]"
    source_str = str(source)
    if not isinstance(entry, dict):
        raise IgnoreFileError(f"{source_str}: {where} must be a mapping")
    unknown = sorted(str(key) for key in entry if key not in _RULE_KEYS)
    if unknown:
        raise IgnoreFileError(
            f"{source_str}: {where} has unknown key(s): {', '.join(unknown)} (allowed: {', '.join(sorted(_RULE_KEYS))})"
        )

    category, check_name = _normalize_check(entry.get("check"), source=source_str, where=where)

    raw_paths = entry.get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise IgnoreFileError(f"{source_str}: {where}.paths must be a non-empty list of path globs")
    paths = tuple(dict.fromkeys(_validate_glob(raw, source=source_str, where=where) for raw in raw_paths))

    reason = entry.get("reason")
    if reason is not None:
        if not isinstance(reason, str):
            raise IgnoreFileError(f"{source_str}: {where}.reason must be a string")
        reason = reason.strip() or None
        if reason is not None and len(reason) > 1024:
            raise IgnoreFileError(f"{source_str}: {where}.reason must be at most 1024 characters")

    expires = _coerce_expires(entry.get("expires"), source=source_str, where=where)

    rule_id = entry.get("id")
    if rule_id is not None:
        if not isinstance(rule_id, str) or not _RULE_ID_PATTERN.fullmatch(rule_id.strip()):
            raise IgnoreFileError(
                f"{source_str}: {where}.id must match [A-Za-z0-9._-] and be no longer than 64 characters"
            )
        rule_id = rule_id.strip()
        if rule_id in used_ids:
            raise IgnoreFileError(f"{source_str}: {where}.id duplicates an earlier rule id: {rule_id!r}")
    else:
        rule_id = f"rule-{index + 1}"
    used_ids.add(rule_id)

    return IgnoreRule(
        rule_id=rule_id,
        index=index,
        category=category,
        check_name=check_name,
        paths=paths,
        reason=reason,
        expires=expires,
        source=source,
    )


def load_ignore_file(path: Path | str) -> list[IgnoreRule]:
    """Parse and validate an ignore document.

    Raises :class:`IgnoreFileError` on any malformed or unsafe content; a
    document that fails to parse suppresses nothing and fails the run rather
    than silently granting exemptions.
    """
    source = Path(path)
    source_str = str(source)
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise IgnoreFileError(f"{source_str}: file is not valid UTF-8") from exc
    except OSError as exc:
        raise IgnoreFileError(f"{source_str}: cannot read ignore file: {exc.strerror or exc}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)  # nosec B506 - pinned SafeLoader
    except yaml.YAMLError as exc:
        raise IgnoreFileError(f"{source_str}: invalid YAML: {_yaml_error_detail(exc)}") from exc
    except RecursionError as exc:
        raise IgnoreFileError(f"{source_str}: invalid YAML: nesting is too deep") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise IgnoreFileError(f"{source_str}: invalid YAML: cannot parse a value: {exc}") from exc

    if not isinstance(raw, dict):
        raise IgnoreFileError(f"{source_str}: top level must be a mapping with an 'ignores' list")
    unknown = sorted(str(key) for key in raw if key not in _TOP_LEVEL_KEYS)
    if unknown:
        raise IgnoreFileError(
            f"{source_str}: unknown top-level key(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(_TOP_LEVEL_KEYS))})"
        )
    version = raw.get("version")
    supported_version = version == 1 or (isinstance(version, str) and version.strip() == "1")
    if version is not None and not supported_version:
        raise IgnoreFileError(f"{source_str}: unsupported version {version!r}; expected 1")
    entries = raw.get("ignores")
    if not isinstance(entries, list) or not entries:
        raise IgnoreFileError(f"{source_str}: 'ignores' must be a non-empty list of rules")

    used_ids: set[str] = set()
    return [
        _rule_from_entry(entry, index=index, source=source, used_ids=used_ids) for index, entry in enumerate(entries)
    ]


def discover_ignore_file(skill_root: Path) -> Path | None:
    """Return ``<skill_root>/skillevaluator-ignore.yaml`` when it is a regular file."""
    candidate = Path(skill_root) / IGNORE_FILE_NAME
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# Safe, skill-relative glob matching
# ---------------------------------------------------------------------------


def _casefold_on_case_insensitive_platform(value: str) -> str:
    return value.casefold() if os.name == "nt" else value


def _posix_anchor(value: str) -> str:
    """Normalize a possible Windows drive prefix to a POSIX-style anchor."""
    if _DRIVE_PATTERN.match(value):
        return "/" + value[2:].lstrip("/")
    return value


def skill_relative_path(file_path: object, skill_root: Path) -> str | None:
    """Return the safe POSIX-style path of *file_path* relative to *skill_root*.

    The comparison is purely lexical (no filesystem access, no symlink
    resolution). Returns ``None`` when the finding has no path, escapes the
    skill root via ``..``, contains empty segments, or cannot be proven to
    live inside the skill root.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    raw = _posix_anchor(file_path.strip().replace("\\", "/"))
    while raw.startswith("./"):
        raw = raw[2:]
    if not raw or raw == ".":
        return None

    root_text = _posix_anchor(Path(skill_root).as_posix())
    root = PurePosixPath(root_text)
    if not root.is_absolute():
        root = PurePosixPath(Path.cwd().as_posix()) / root

    if raw.startswith("/"):
        try:
            relative = PurePosixPath(raw).relative_to(root)
        except ValueError:
            return None
        segments = relative.parts
    else:
        segments = tuple(raw.split("/"))

    if not segments or any(segment in {"", ".", ".."} for segment in segments):
        return None
    return "/".join(segments)


def _compile_glob(pattern: str) -> list[re.Pattern[str] | None]:
    """Compile a validated glob to per-segment regexes (``None`` marks ``**``)."""
    compiled: list[re.Pattern[str] | None] = []
    for segment in pattern.split("/"):
        if segment == "**":
            compiled.append(None)
        else:
            source = _casefold_on_case_insensitive_platform(segment)
            compiled.append(re.compile(fnmatch.translate(source)))
    return compiled


def glob_matches(pattern: str, relative_path: str) -> bool:
    """Match a validated glob against a canonical skill-relative path.

    ``*``/``?``/``[...]`` stay within one path segment; ``**`` as its own
    segment spans any number of segments. Case follows the host platform.
    """
    glob_segments = _compile_glob(pattern)
    path_segments = [_casefold_on_case_insensitive_platform(segment) for segment in relative_path.split("/")]

    def match(glob_index: int, path_index: int) -> bool:
        while glob_index < len(glob_segments):
            segment = glob_segments[glob_index]
            if segment is None:  # '**'
                if glob_index == len(glob_segments) - 1:
                    return True
                return any(match(glob_index + 1, consumed) for consumed in range(path_index, len(path_segments) + 1))
            if path_index >= len(path_segments):
                return False
            if segment.fullmatch(path_segments[path_index]) is None:
                return False
            glob_index += 1
            path_index += 1
        return path_index == len(path_segments)

    return match(0, 0)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@dataclass
class _RuleState:
    rule: IgnoreRule
    expired: bool
    matched_findings: list[dict[str, Any]] = field(default_factory=list)


def _severity_value(severity: object) -> str:
    value = getattr(severity, "value", None)
    return str(value if value is not None else severity).lower()


def _finding_check_id(category: object, check_name: object) -> str:
    return f"{str(category).upper()}.{check_name}"


def get_suppression_ledger(results: list[ValidationResult]) -> dict[str, Any] | None:
    """Return the run-wide suppression ledger recorded on any result."""
    for result in results:
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            ledger = metadata.get(SUPPRESSION_LEDGER_KEY)
            if isinstance(ledger, dict):
                return ledger
    return None


def suppressed_finding_records(result: ValidationResult) -> list[dict[str, Any]]:
    """Return suppressed finding records attached to one result, if any."""
    metadata = getattr(result, "metadata", None)
    if not isinstance(metadata, dict):
        return []
    records = metadata.get(SUPPRESSED_FINDINGS_KEY)
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def _blocked_reason(result: ValidationResult, category: str) -> str | None:
    if category.upper() in _NON_EXEMPTABLE_CATEGORIES:
        return "secret findings cannot be exempted"
    if result.is_incomplete:
        scanners = ", ".join(result.incomplete_scans) or "a scanner"
        return f"required evidence is incomplete ({scanners} did not complete)"
    return None


def apply_ignore_rules(
    results: list[ValidationResult],
    rules: list[IgnoreRule],
    *,
    source: Path,
    discovered: bool,
    skill_root: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply exemption rules to validated results and record a full ledger.

    Matched, unexpired, exemptable findings are removed from each result's
    gate (its pass/fail state is recalculated) but retained with full rule
    provenance in ``metadata['suppressed_findings']``. A structured ledger is
    stamped onto every result under ``metadata['suppression_ledger']`` so all
    reporters render the same suppressed counts and warnings.
    """
    today = (now or datetime.now(tz=UTC)).date()
    states = [_RuleState(rule=rule, expired=rule.is_expired(today)) for rule in rules]

    warnings: list[str] = []
    blocked: list[dict[str, Any]] = []

    def warn(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    for state in states:
        if state.expired:
            warn(
                f"ignore rule {state.rule.rule_id} ({state.rule.check_id}) expired on "
                f"{state.rule.expires.isoformat()} and no longer suppresses findings"
            )

    active = [state for state in states if not state.expired]
    blocked_rule_ids: set[str] = set()

    for result in results:
        records: list[dict[str, Any]] = []
        if result.findings:
            surviving = list(result.findings)
            for finding in tuple(result.findings):
                check_id = _finding_check_id(finding.category, finding.check_name)
                relative = skill_relative_path(finding.file_path, skill_root)
                for state in active:
                    rule = state.rule
                    if rule.check_id != check_id:
                        continue
                    reason = _blocked_reason(result, str(finding.category).upper())
                    if reason is not None:
                        entry = {
                            "rule_id": rule.rule_id,
                            "check": rule.check_id,
                            "file_path": finding.file_path,
                            "line_number": finding.line_number,
                            "reason": reason,
                        }
                        blocked.append(entry)
                        blocked_rule_ids.add(rule.rule_id)
                        warn(
                            f"ignore rule {rule.rule_id} ({rule.check_id}) was not applied to "
                            f"{finding.location or 'a finding'}: {reason}"
                        )
                        break
                    if relative is None or not any(glob_matches(path, relative) for path in rule.paths):
                        continue
                    matched_glob = next(path for path in rule.paths if glob_matches(path, relative))
                    record = {
                        "rule_id": rule.rule_id,
                        "check": rule.check_id,
                        "reason": rule.reason,
                        "expires": rule.expires.isoformat() if rule.expires is not None else None,
                        "source": str(source),
                        "discovered": discovered,
                        "matched_glob": matched_glob,
                        "matched_path": relative,
                        "validator": result.validator_name,
                        "category": finding.category,
                        "severity": _severity_value(finding.severity),
                        "check_name": finding.check_name,
                        "message": finding.message,
                        "file_path": finding.file_path,
                        "line_number": finding.line_number,
                        "location": finding.location,
                    }
                    state.matched_findings.append(record)
                    records.append(record)
                    surviving.remove(finding)
                    break
            if len(surviving) != len(result.findings):
                result.findings = surviving
                result.recalculate_from_findings()
        if records:
            result.metadata[SUPPRESSED_FINDINGS_KEY] = [*suppressed_finding_records(result), *records]

    for state in active:
        if not state.matched_findings and state.rule.rule_id not in blocked_rule_ids:
            warn(
                f"ignore rule {state.rule.rule_id} ({state.rule.check_id}) never matched a finding; review or remove it"
            )

    rule_entries: list[dict[str, Any]] = []
    suppressed_count = 0
    for state in states:
        matched = state.matched_findings
        suppressed_count += len(matched)
        entry = state.rule.to_dict()
        entry.update(
            {
                "status": "expired" if state.expired else "active",
                "matched_count": len(matched),
                "findings": matched,
            }
        )
        rule_entries.append(entry)

    ledger: dict[str, Any] = {
        "source": str(source),
        "discovered": discovered,
        "rule_count": len(states),
        "active_rule_count": len(active),
        "expired_rule_count": sum(1 for state in states if state.expired),
        "suppressed_count": suppressed_count,
        "rules": rule_entries,
        "blocked": blocked,
        "warnings": warnings,
    }
    for result in results:
        result.metadata[SUPPRESSION_LEDGER_KEY] = ledger
    return ledger


__all__ = [
    "IGNORE_FILE_NAME",
    "SUPPRESSED_FINDINGS_KEY",
    "SUPPRESSION_LEDGER_KEY",
    "IgnoreFileError",
    "IgnoreRule",
    "apply_ignore_rules",
    "discover_ignore_file",
    "get_suppression_ledger",
    "glob_matches",
    "load_ignore_file",
    "skill_relative_path",
    "suppressed_finding_records",
]
