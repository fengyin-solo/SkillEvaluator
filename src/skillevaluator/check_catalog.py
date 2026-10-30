# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative Tier 1 check catalog shared by the runtime and the ``checks`` command.

This module is the single source of truth for which Tier 1 checks exist, which
run by default, which content types they apply to, which optional packages or
external executables they rely on, and which structured evidence categories
they emit. The validation pipeline in :mod:`skillevaluator.tier1.commands`
derives its check ordering, default/opt-in split, and content-type applicability
from the registry declared here, so the ``checks`` command can never silently
list a different lineup than the one that actually runs.

Availability probing is strictly offline:

* external executables are located with ``shutil.which`` (never executed), and
* optional Python packages are detected with :func:`importlib.util.find_spec`
  (never imported).

No scanner is launched and no model/provider request is made when building the
catalog, so the command stays read-only, fast, and CI/offline friendly.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from skillevaluator.cli_core import (
    _detect_from_directory,
    _detect_from_file,
    _detect_from_nested_structure,
    _detect_from_path_parts,
)
from skillevaluator.constants import (
    CONTENT_TYPE_PLUGIN,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_UNKNOWN,
    CONTENT_TYPE_WORKFLOWS,
)
from skillevaluator.utils.tool_runner import Tools
from skillevaluator.validators.code_risk import CodeRiskValidator
from skillevaluator.validators.dependencies import DependencySecurityValidator
from skillevaluator.validators.hygiene import HygieneValidator
from skillevaluator.validators.license import LicenseValidator
from skillevaluator.validators.plugin_schema import PluginSchemaValidator
from skillevaluator.validators.quality_score import QualityScoreValidator
from skillevaluator.validators.rules_schema import RulesSchemaValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.script_lint import ScriptLintValidator
from skillevaluator.validators.secrets import SecretsValidator
from skillevaluator.validators.security import SecurityValidator
from skillevaluator.validators.unicode_smuggle import UnicodeSmuggleValidator
from skillevaluator.validators.version import VersionValidator
from skillevaluator.validators.workflows_schema import WorkflowsSchemaValidator

if TYPE_CHECKING:
    from pathlib import Path

    from skillevaluator.validators.base import ValidatorBase

# Content types a check can be declared applicable to. ``unknown`` is handled
# separately because the runtime treats an unclassifiable target as skill-like.
KNOWN_CONTENT_TYPES: tuple[str, ...] = (
    CONTENT_TYPE_SKILL,
    CONTENT_TYPE_RULES,
    CONTENT_TYPE_WORKFLOWS,
    CONTENT_TYPE_PLUGIN,
)
CONTENT_TYPE_CHOICES: tuple[str, ...] = KNOWN_CONTENT_TYPES

# Activation modes. ``default`` checks run unless ``--checks`` narrows the run;
# ``opt-in`` checks are recognized but excluded from the default lineup.
ACTIVATION_DEFAULT = "default"
ACTIVATION_OPT_IN = "opt-in"

# Requirement kinds. New kinds may be added in the future; consumers must treat
# an unrecognized kind as an opaque dependency rather than failing.
REQUIREMENT_EXECUTABLE = "executable"
REQUIREMENT_PACKAGE = "package"

# How a requirement's availability was probed. Both probes are offline.
PROBE_PATH = "path"
PROBE_IMPORT = "import"

CHECKS_CATALOG_SCHEMA_VERSION = 1

# Schema validator selected for each (forced or detected) content type. This is
# the same mapping ``run_validation`` uses; kept here so the catalog reports the
# validator the runtime would actually instantiate for a given target.
_SCHEMA_VALIDATORS: dict[str, type[ValidatorBase]] = {
    CONTENT_TYPE_RULES: RulesSchemaValidator,
    CONTENT_TYPE_WORKFLOWS: WorkflowsSchemaValidator,
    CONTENT_TYPE_PLUGIN: PluginSchemaValidator,
}
_SCHEMA_FALLBACK_VALIDATOR: type[ValidatorBase] = SchemaValidator


def _module_installed(module_name: str) -> bool:
    """Return whether a module is importable without importing it (offline)."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _validator_name(cls: type[ValidatorBase]) -> str:
    """Read a validator's display name from a throwaway instance.

    Construction uses the same cheap configuration the runtime builds before a
    scan; no file tree is walked, no tool is run, and no model is called. Any
    unexpected construction failure falls back to the class name so the catalog
    remains available.
    """
    try:
        return str(cls().name)
    except Exception:  # pragma: no cover - defensive: catalog must stay available
        return cls.__name__


@dataclass(frozen=True)
class CheckRequirement:
    """An optional package or external executable a check depends on.

    Attributes:
        kind: ``executable`` for a CLI binary located on PATH, ``package`` for
            an importable Python extra.
        name: Canonical executable command or import/display name.
        description: What the dependency contributes to the check.
        probe: Offline probe used to resolve ``available``/``installed``.
        required: True when the check cannot produce complete evidence without
            this dependency; False when the check degrades gracefully without it.
        bundled_extra: The pip extra that bundles this dependency, or None when
            it must be installed independently of SkillEvaluator.
        enabled_by: The flag/capability that switches an otherwise-inactive
            dependency on (e.g. ``--llm``), if it is off by default.
        tool_attribute: Attribute name on
            :class:`skillevaluator.utils.tool_runner.Tools` for executables.
        import_name: Importable module probed with ``find_spec`` for packages.
        install_hint: Short installation guidance.
    """

    kind: str
    name: str
    description: str
    probe: str
    required: bool = False
    bundled_extra: str | None = None
    enabled_by: str | None = None
    tool_attribute: str | None = None
    import_name: str | None = None
    install_hint: str | None = None

    def _availability(self) -> bool:
        """Resolve availability without executing a binary or importing a module."""
        if self.kind == REQUIREMENT_EXECUTABLE and self.tool_attribute:
            return bool(getattr(Tools, self.tool_attribute).is_available)
        if self.kind == REQUIREMENT_PACKAGE and self.import_name:
            return _module_installed(self.import_name)
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize with a stable, forward-compatible key order."""
        available = self._availability()
        payload: dict[str, Any] = {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "required": self.required,
            "bundled_extra": self.bundled_extra,
            "enabled_by": self.enabled_by,
            "probe": self.probe,
        }
        # ``available`` for executables and ``installed`` for packages both
        # answer "is the dependency usable right now", using offline probes.
        payload["available" if self.kind == REQUIREMENT_EXECUTABLE else "installed"] = available
        if self.install_hint:
            payload["install_hint"] = self.install_hint
        return payload


@dataclass(frozen=True)
class CheckSpec:
    """One Tier 1 check in the authoritative registry.

    Attributes:
        id: Canonical check identifier used by ``--checks`` and publication
            evidence.
        display_name: Stable human-facing name for the check.
        description: One-line description of what the check does.
        activation: ``default`` or ``opt-in``.
        skill_only: True when the runtime runs this check only for skill-like
            targets (skills and unclassifiable targets) and skips it for rules,
            workflows, and plugins.
        evidence_categories: Structured ``Finding.category`` values the check
            primarily emits.
        validator_classes: Validator classes the runtime instantiates for this
            check. Display names are read from live instances, never duplicated
            as handwritten strings.
        requirements: Optional packages/external executables the check uses.
    """

    id: str
    display_name: str
    description: str
    activation: str
    skill_only: bool
    evidence_categories: tuple[str, ...]
    validator_classes: tuple[type[ValidatorBase], ...] = field(default_factory=tuple)
    requirements: tuple[CheckRequirement, ...] = field(default_factory=tuple)

    @property
    def enabled_by_default(self) -> bool:
        """Whether the check runs in the default (un-narrowed) lineup."""
        return self.activation == ACTIVATION_DEFAULT

    def applicable_content_types(self) -> list[str]:
        """Known content types for which the runtime runs this check."""
        if self.skill_only:
            return [CONTENT_TYPE_SKILL]
        return list(KNOWN_CONTENT_TYPES)

    def applies_to(self, content_type: str | None) -> bool:
        """Mirror the runtime applicability gate in ``run_validation``.

        The pipeline treats ``None`` and ``unknown`` as skill-like, so
        skill-only checks still run for unclassifiable targets.
        """
        if not self.skill_only:
            return True
        return content_type in (None, CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN)

    def component_names(self, content_type: str | None = None) -> list[str]:
        """Return the live validator display names backing this check.

        Names come from instantiating the same validator classes the runtime
        uses, so the catalog cannot advertise a name the pipeline does not use.
        """
        if self.id == "schema":
            return self._schema_component_names(content_type)
        return [_validator_name(cls) for cls in self.validator_classes if cls is not None]

    @staticmethod
    def _schema_component_names(content_type: str | None) -> list[str]:
        if content_type in _SCHEMA_VALIDATORS:
            return [_validator_name(_SCHEMA_VALIDATORS[content_type])]
        if content_type in (CONTENT_TYPE_SKILL, CONTENT_TYPE_UNKNOWN):
            # Skill and unclassifiable targets fall back to the skill schema.
            return [_validator_name(_SCHEMA_FALLBACK_VALIDATOR)]
        # Unfiltered catalog: show every schema validator the runtime may pick.
        names = [_validator_name(_SCHEMA_FALLBACK_VALIDATOR)]
        names.extend(_validator_name(cls) for cls in _SCHEMA_VALIDATORS.values())
        return names

    def to_dict(
        self,
        *,
        applicable: bool | None = None,
        resolved_content_type: str | None = None,
    ) -> dict[str, Any]:
        """Serialize with a stable key order; extra keys remain allowed."""
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.display_name,
            "description": self.description,
            "enabled_by_default": self.enabled_by_default,
            "activation": self.activation,
            "applicable_content_types": self.applicable_content_types(),
            "applies_to_unknown": self.applies_to(CONTENT_TYPE_UNKNOWN),
            "evidence_categories": list(self.evidence_categories),
            "components": self.component_names(resolved_content_type),
            "requirements": [requirement.to_dict() for requirement in self.requirements],
        }
        # Only present when the catalog is filtered for a concrete target so
        # the all-checks listing does not imply a specific applicability.
        if applicable is not None:
            payload["applicable"] = applicable
        return payload


def _executable(
    name: str,
    *,
    tool_attribute: str,
    description: str,
    required: bool,
    bundled_extra: str | None,
) -> CheckRequirement:
    return CheckRequirement(
        kind=REQUIREMENT_EXECUTABLE,
        name=name,
        description=description,
        probe=PROBE_PATH,
        required=required,
        bundled_extra=bundled_extra,
        tool_attribute=tool_attribute,
        install_hint=getattr(Tools, tool_attribute).get_install_hint(),
    )


def _llm_requirement(description: str, *, enabled_by: str) -> CheckRequirement:
    return CheckRequirement(
        kind=REQUIREMENT_PACKAGE,
        name="llm",
        description=description,
        probe=PROBE_IMPORT,
        required=False,
        bundled_extra="llm",
        enabled_by=enabled_by,
        import_name="openai",
        install_hint=(
            "Install the LLM extra: uv tool install skillevaluator[llm] "
            "(a provider credential is also required)."
        ),
    )


# The ordered Tier 1 registry. Order is the canonical pipeline order used by
# ``run_validation`` and ``enabled_check_lineup``. ``tier1.commands`` derives
# DEFAULT_CHECKS / OPTIONAL_CHECKS from this table.
CHECK_REGISTRY: tuple[CheckSpec, ...] = (
    CheckSpec(
        id="schema",
        display_name="Schema & Structure",
        description="Validate the manifest frontmatter, required fields, and repository structure.",
        activation=ACTIVATION_DEFAULT,
        skill_only=False,
        evidence_categories=("SCHEMA", "PLUGIN_SCHEMA"),
        validator_classes=(
            SchemaValidator,
            RulesSchemaValidator,
            WorkflowsSchemaValidator,
            PluginSchemaValidator,
        ),
    ),
    CheckSpec(
        id="version",
        display_name=_validator_name(VersionValidator),
        description="Validate optional metadata.version labels and require strictly increasing bumps.",
        activation=ACTIVATION_DEFAULT,
        skill_only=True,
        evidence_categories=("VERSION",),
        validator_classes=(VersionValidator,),
    ),
    CheckSpec(
        id="security",
        display_name="Security Scan",
        description="Detect security vulnerabilities with the external skillspector scanner.",
        activation=ACTIVATION_DEFAULT,
        skill_only=False,
        evidence_categories=("SECURITY",),
        validator_classes=(SecurityValidator,),
        requirements=(
            _executable(
                "skillspector",
                tool_attribute="skillspector",
                description="External static security scanner that produces the SECURITY evidence.",
                required=True,
                bundled_extra=None,
            ),
            _llm_requirement(
                "Optional LLM-backed security analysis on top of skillspector.",
                enabled_by="--llm",
            ),
            _llm_requirement(
                "Optional second LLM pass that suppresses false-positive findings.",
                enabled_by="--llm-verify",
            ),
        ),
    ),
    CheckSpec(
        id="pii",
        display_name="PII Scan",
        description="Detect PII and local identifiers (home paths, emails, keys) with built-in patterns.",
        activation=ACTIVATION_DEFAULT,
        skill_only=False,
        evidence_categories=("PII",),
        validator_classes=(SecurityValidator,),
        requirements=(
            _llm_requirement(
                "Optional LLM verification of candidate PII findings.",
                enabled_by="--llm-verify",
            ),
        ),
    ),
    CheckSpec(
        id="license",
        display_name=_validator_name(LicenseValidator),
        description="Validate license files, SPDX identifiers, and source-file license headers.",
        activation=ACTIVATION_DEFAULT,
        skill_only=False,
        evidence_categories=("LICENSE",),
        validator_classes=(LicenseValidator,),
    ),
    CheckSpec(
        id="code-integrity",
        display_name="Code Integrity",
        description="Static code risk, hardcoded-secret detection, and code hygiene checks.",
        activation=ACTIVATION_DEFAULT,
        skill_only=False,
        evidence_categories=("BANDIT", "SEMGREP", "SECRET"),
        validator_classes=(CodeRiskValidator, SecretsValidator, HygieneValidator),
        requirements=(
            _executable(
                "bandit",
                tool_attribute="bandit",
                description="Python security linter producing BANDIT findings.",
                required=False,
                bundled_extra="security",
            ),
            _executable(
                "semgrep",
                tool_attribute="semgrep",
                description="Packaged high-confidence Semgrep rules producing SEMGREP findings.",
                required=False,
                bundled_extra=None,
            ),
            _executable(
                "gitleaks",
                tool_attribute="gitleaks",
                description="Secret scanner producing SECRET findings.",
                required=False,
                bundled_extra=None,
            ),
        ),
    ),
    CheckSpec(
        id="dependency",
        display_name=_validator_name(DependencySecurityValidator),
        description="Audit Python dependencies for known CVEs with pip-audit and optional Safety.",
        activation=ACTIVATION_OPT_IN,
        skill_only=False,
        evidence_categories=("CVE", "SAFETY"),
        validator_classes=(DependencySecurityValidator,),
        requirements=(
            _executable(
                "pip-audit",
                tool_attribute="pip_audit",
                description="Primary dependency auditor producing CVE findings against PyPI/OSV.",
                required=True,
                bundled_extra="security",
            ),
            _executable(
                "safety",
                tool_attribute="safety",
                description="Optional secondary advisory database producing SAFETY findings.",
                required=False,
                bundled_extra=None,
            ),
        ),
    ),
    CheckSpec(
        id="unicode",
        display_name=_validator_name(UnicodeSmuggleValidator),
        description="Detect invisible Unicode characters and ASCII smuggling/trojan-source patterns.",
        activation=ACTIVATION_DEFAULT,
        skill_only=False,
        evidence_categories=("UNICODE",),
        validator_classes=(UnicodeSmuggleValidator,),
    ),
    CheckSpec(
        id="quality",
        display_name=_validator_name(QualityScoreValidator),
        description="Deterministic four-dimension skill quality scoring with a configurable gate.",
        activation=ACTIVATION_DEFAULT,
        skill_only=True,
        evidence_categories=("QUALITY",),
        validator_classes=(QualityScoreValidator,),
    ),
    CheckSpec(
        id="lint",
        display_name=_validator_name(ScriptLintValidator),
        description="Advisory AST-based code quality checks for skill scripts.",
        activation=ACTIVATION_DEFAULT,
        skill_only=True,
        evidence_categories=("SCRIPT_LINT",),
        validator_classes=(ScriptLintValidator,),
    ),
)

# Fast lookup by canonical id.
CHECK_SPECS: dict[str, CheckSpec] = {spec.id: spec for spec in CHECK_REGISTRY}

# Derived lineup constants consumed by the runtime pipeline.
DEFAULT_CHECKS: tuple[str, ...] = tuple(
    spec.id for spec in CHECK_REGISTRY if spec.activation == ACTIVATION_DEFAULT
)
OPTIONAL_CHECKS: tuple[str, ...] = tuple(
    spec.id for spec in CHECK_REGISTRY if spec.activation == ACTIVATION_OPT_IN
)
RECOGNIZED_CHECKS: frozenset[str] = frozenset(DEFAULT_CHECKS) | frozenset(OPTIONAL_CHECKS)

# Check aliases accepted by ``--checks`` (resolved before lookup).
CHECK_ALIASES: dict[str, str] = {
    "code": "code-integrity",
    "code-risk": "code-integrity",
    "scripts": "lint",
    "script-lint": "lint",
    "dependencies": "dependency",
    "deps": "dependency",
    "dependency-audit": "dependency",
    "licence": "license",
    "license-check": "license",
}


def schema_validator_for(content_type: str | None, policy: Any = None) -> ValidatorBase:
    """Return the schema validator selected for a content type (runtime parity).

    Rules, workflows, and plugins use dedicated validators; skill and unknown
    content fall back to the skill :class:`SchemaValidator`.
    """
    cls = _SCHEMA_VALIDATORS.get(content_type or "", _SCHEMA_FALLBACK_VALIDATOR)
    if cls is SchemaValidator:
        return SchemaValidator(policy=policy)
    return cls()


def get_check_spec(check_id: str) -> CheckSpec | None:
    """Return the spec for a canonical check id (alias resolution is separate)."""
    return CHECK_SPECS.get(check_id)


def check_applies_to(check_id: str, content_type: str | None) -> bool:
    """Return whether the runtime runs ``check_id`` for a content type."""
    spec = CHECK_SPECS.get(check_id)
    return spec.applies_to(content_type) if spec is not None else False


def resolve_check_alias(raw: str) -> str:
    """Normalize one user-supplied check name to its canonical id."""
    return CHECK_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def explain_unknown_content_type(path: Path) -> str:
    """Explain why content-type detection could not classify ``path``.

    Mirrors the detection stages in :func:`skillevaluator.cli_core.detect_content_type`
    so the reason tracks the real detector instead of a handwritten guess.
    """
    reasons: list[str] = []
    if not path.exists():
        return "target path does not exist"
    if path.is_file() and _detect_from_file(path) is None:
        reasons.append(
            "file is not SKILL.md, a .mdc rule, workflow-rules.mdc, an agent_plugin.yaml/.yml "
            "manifest, or .claude-plugin/plugin.json"
        )
    if path.is_dir() and _detect_from_directory(path) is None:
        reasons.append(
            "directory contains no SKILL.md, workflow-rules.mdc, *.mdc rule file, agent_plugin.yaml/.yml, "
            "or .claude-plugin/plugin.json manifest"
        )
    if _detect_from_path_parts(path) is None:
        reasons.append("path has no skills/, team-skills/, team-rules/, workflows/, or team-workflows/ component")
    if path.is_dir() and _detect_from_nested_structure(path) is None:
        reasons.append("no nested skills/, team-skills/, team-rules/, workflows/, or team-workflows/ tree was found")
    if not reasons:
        return "content type could not be determined from the target"
    return "; ".join(reasons)


def _resolve_target(
    *,
    content_type: str | None,
    target_path: Path | None,
) -> tuple[str | None, str | None, str | None]:
    """Resolve the effective content type for filtering.

    Returns ``(content_type, source, reason)`` where source is ``"flag"`` when
    forced via ``--type``, ``"auto"`` when detected from a path, or ``None``
    when no filter applies. ``reason`` is set only for an ``unknown`` detection.
    """
    if content_type is not None:
        return content_type, "flag", None
    if target_path is not None:
        from skillevaluator.cli_core import detect_content_type

        detected = detect_content_type(target_path)
        if detected == CONTENT_TYPE_UNKNOWN:
            return CONTENT_TYPE_UNKNOWN, "auto", explain_unknown_content_type(target_path)
        return detected, "auto", None
    return None, None, None


def build_check_catalog(
    *,
    content_type: str | None = None,
    target_path: Path | None = None,
    only_applicable: bool = False,
) -> dict[str, Any]:
    """Build the serializable check catalog.

    When ``content_type`` (``--type``) or ``target_path`` (auto-detection) is
    supplied, each check carries an ``applicable`` flag resolved with the same
    content-type gate the runtime pipeline uses. ``only_applicable`` drops
    non-applicable checks from the listing entirely.
    """
    resolved_type, type_source, unknown_reason = _resolve_target(
        content_type=content_type,
        target_path=target_path,
    )

    checks: list[dict[str, Any]] = []
    for spec in CHECK_REGISTRY:
        applicable: bool | None = None
        if resolved_type is not None:
            applicable = spec.applies_to(resolved_type)
        if only_applicable and applicable is False:
            continue
        entry = spec.to_dict(
            applicable=applicable,
            resolved_content_type=resolved_type,
        )
        checks.append(entry)

    return {
        "schema_version": 1,
        "command": "checks",
        "tier": 1,
        "content_type": resolved_type,
        "content_type_source": type_source,
        "content_type_reason": unknown_reason,
        "target_path": str(target_path) if target_path is not None else None,
        "availability_probe": "offline",
        "checks": checks,
    }


def render_checks_text(catalog: dict[str, Any]) -> str:
    """Render the stable human-readable listing used by ``checks`` without JSON."""
    checks: Sequence[dict[str, Any]] = catalog["checks"]
    resolved_type = catalog.get("content_type")
    lines: list[str] = []

    default_count = sum(1 for check in checks if check["enabled_by_default"])
    opt_in_count = len(checks) - default_count
    lines.append(f"Tier 1 checks ({len(checks)}): {default_count} default, {opt_in_count} opt-in")
    lines.append(
        "Availability probe: offline (PATH/import lookup only; no scanners or model requests are run)"
    )
    if resolved_type is not None:
        lines.append(f"Content type: {resolved_type} (source: {catalog.get('content_type_source')})")
        if resolved_type == CONTENT_TYPE_UNKNOWN and catalog.get("content_type_reason"):
            lines.append(f"  reason: {catalog['content_type_reason']}")
    else:
        lines.append("Content type: not filtered (pass --type or a PATH to resolve applicability)")
    lines.append("")

    headers = ("ID", "MODE", "APPLIES TO", "NAME")
    rows = [headers]
    for check in checks:
        mode = "default" if check["enabled_by_default"] else "opt-in"
        applies = ",".join(check["applicable_content_types"])
        if check.get("applies_to_unknown"):
            applies += ",unknown*"
        name = check["name"]
        if resolved_type is not None and check.get("applicable") is False:
            name += "  [not applicable]"
        rows.append((check["id"], mode, applies, name))

    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    for index, row in enumerate(rows):
        lines.append(
            f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]:<{widths[2]}}  {row[3]}"
        )
        if index == 0:
            lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))

    # Dependencies and their live availability, grouped per check.
    requirement_lines: list[str] = []
    for check in checks:
        requirements = check.get("requirements") or []
        if not requirements:
            continue
        statuses = []
        for requirement in requirements:
            state = "yes" if requirement.get("available", requirement.get("installed")) else "no"
            detail = "" if requirement.get("required") else " (optional"
            if requirement.get("enabled_by"):
                detail += f", {requirement['enabled_by']}" if detail else f" ({requirement['enabled_by']}"
            if detail:
                detail += ")"
            if requirement.get("bundled_extra"):
                bundle = f", bundled: skillevaluator[{requirement['bundled_extra']}]"
            else:
                bundle = ", external install"
            statuses.append(f"{requirement['name']}={state}{detail}{bundle}")
        requirement_lines.append(f"  {check['id']}: " + "; ".join(statuses))
    if requirement_lines:
        lines.append("")
        lines.append("Dependencies:")
        lines.extend(requirement_lines)
        lines.append("")
        lines.append("* unknown targets are validated with the skill lineup (skill fallback).")

    return "\n".join(lines)


def run_checks_command(
    *,
    target_path: Path | None,
    content_type: str | None,
    as_json: bool,
    only_applicable: bool,
) -> int:
    """Entry point for the read-only ``checks`` command.

    Returns a process exit code. The command never validates content, runs an
    external scanner, or calls a model; it only resolves the catalog and probes
    executable/package availability offline.
    """
    import json

    import click

    catalog = build_check_catalog(
        content_type=content_type,
        target_path=target_path,
        only_applicable=only_applicable,
    )
    if as_json:
        # Stable key order: dicts are already inserted in the declared order;
        # sort_keys is intentionally NOT used so field order stays documented.
        click.echo(json.dumps(catalog, ensure_ascii=False, indent=2))
    else:
        click.echo(render_checks_text(catalog))
    return 0

