# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for deterministic quality scoring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypedDict, cast

import yaml

from skillevaluator.constants import QUALITY_DIMENSIONS, QUALITY_SCORE_WEIGHTS

DEFAULT_QUALITY_CONFIG = files("skillevaluator.config").joinpath("quality-default.yaml")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")
_TOP_LEVEL_KEYS = frozenset({"expected_headers", "dimension_weights", "dimension_minimums"})
_EXPECTED_HEADER_KEYS = frozenset({"heading", "alternatives", "points", "message", "suggestion"})
_DIMENSION_KEYS = frozenset(QUALITY_DIMENSIONS)
# Decimal weights like 0.1 + 0.2 + 0.3 + 0.4 accumulate a 1e-16 float error, so
# the sum-to-one check accepts this narrow band instead of rejecting valid
# configs (or accepting genuinely wrong totals).
_WEIGHT_SUM_TOLERANCE = 1e-9


class QualityConfigError(ValueError):
    """Raised when a quality configuration is invalid."""


class ExpectedHeader(TypedDict):
    """A Markdown heading expected by the quality validator."""

    heading: str
    alternatives: list[str]
    points: int | float
    message: str
    suggestion: str


class QualityConfig(TypedDict):
    """Validated deterministic quality configuration.

    ``dimension_weights`` always names all four dimensions (defaults are filled
    in when the file omits the block); ``dimension_minimums`` holds only the
    thresholds the file actually set.
    """

    expected_headers: list[ExpectedHeader]
    dimension_weights: dict[str, float]
    dimension_minimums: dict[str, float]


class QualityConfigProvenance(TypedDict):
    """Which quality configuration produced a score, as recorded in reports."""

    source: Literal["packaged", "custom"]
    file: str | None
    digest: str


def _normalize_heading(value: object, source: str, location: str) -> str:
    heading_match = _ATX_HEADING_RE.fullmatch(value) if isinstance(value, str) else None
    heading_title = heading_match.group(2).strip() if heading_match else ""
    if heading_match is None or not heading_title:
        raise QualityConfigError(f"{source}: {location} must be a single ATX Markdown heading")
    return f"{heading_match.group(1)} {heading_title}"


def _reject_unknown_keys(mapping: dict, allowed: frozenset[str], source: str, location: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise QualityConfigError(
            f"{source}: unknown key(s) in {location}: {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})"
        )


def _is_finite_number(value: object) -> bool:
    """Whether ``value`` is a finite YAML number (booleans excluded)."""
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:  # isfinite() on an int too large for a float
        return False


def _dimension_block(raw: object, source: str, block: str) -> dict:
    """Return the raw dimension block as a mapping, with a clear error otherwise."""
    if not isinstance(raw, dict):
        raise QualityConfigError(f"{source}: {block} must be a mapping of dimension names to numbers")
    _reject_unknown_keys(raw, _DIMENSION_KEYS, source, block)
    return raw


def _normalize_dimension_weights(raw: object, source: str) -> dict[str, float]:
    """Validate the optional ``dimension_weights`` block, filling defaults when absent."""
    if raw is None:
        return dict(QUALITY_SCORE_WEIGHTS)
    mapping = _dimension_block(raw, source, "dimension_weights")

    missing = [name for name in QUALITY_DIMENSIONS if name not in mapping]
    if missing:
        raise QualityConfigError(
            f"{source}: dimension_weights must set all four dimensions; missing: {', '.join(missing)}"
        )

    weights: dict[str, float] = {}
    for name in QUALITY_DIMENSIONS:
        value = mapping[name]
        if not _is_finite_number(value):
            raise QualityConfigError(f"{source}: dimension_weights.{name} must be a finite non-negative number")
        number = float(value)
        if number < 0:
            raise QualityConfigError(f"{source}: dimension_weights.{name} must be a finite non-negative number")
        weights[name] = number

    # math.fsum sums in a deterministic, error-compensating order so equivalent
    # configs cannot land on different sides of the sum-to-one check.
    total = math.fsum(weights[name] for name in QUALITY_DIMENSIONS)
    if not math.isfinite(total) or abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise QualityConfigError(f"{source}: dimension_weights must sum to 1.0 (got {format(total, '.12g')})")
    return weights


def _normalize_dimension_minimums(raw: object, source: str) -> dict[str, float]:
    """Validate the optional ``dimension_minimums`` block; any subset is allowed."""
    if raw is None:
        return {}
    mapping = _dimension_block(raw, source, "dimension_minimums")

    minimums: dict[str, float] = {}
    for name in QUALITY_DIMENSIONS:
        if name not in mapping:
            continue
        value = mapping[name]
        if not _is_finite_number(value):
            raise QualityConfigError(f"{source}: dimension_minimums.{name} must be a number between 0 and 100")
        number = float(value)
        if not 0 <= number <= 100:
            raise QualityConfigError(f"{source}: dimension_minimums.{name} must be a number between 0 and 100")
        minimums[name] = number
    return minimums


def _freeze(value: Any) -> Any:
    """Recursively turn mappings into read-only mapping proxies."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_freeze(item) for item in value]
    return value


def _to_plain(value: Any) -> Any:
    """Turn read-only mappings back into plain dicts for JSON serialization."""
    if isinstance(value, Mapping):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader that rejects repeated mapping keys instead of keeping the last value."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # unhashable keys are rejected by the base constructor
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None)
    if mark is None or problem is None:
        return str(exc)
    return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"


def load_quality_config(path: Path | None = None) -> QualityConfig:
    source = str(path) if path else "packaged quality-default.yaml"
    try:
        text = path.read_text(encoding="utf-8") if path else DEFAULT_QUALITY_CONFIG.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise QualityConfigError(f"{source}: file is not valid UTF-8") from exc
    except OSError as exc:
        raise QualityConfigError(f"{source}: cannot read file: {exc.strerror or exc}") from exc
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)  # nosec B506
    except yaml.YAMLError as exc:
        raise QualityConfigError(f"{source}: invalid YAML: {_yaml_error_detail(exc)}") from exc
    except RecursionError as exc:
        raise QualityConfigError(f"{source}: invalid YAML: nesting is too deep") from exc
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise QualityConfigError(f"{source}: invalid YAML: cannot parse a value: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("expected_headers"), list):
        raise QualityConfigError(f"{source}: expected_headers must be a list")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, source, "top level")

    # Parse and validate the dimension settings before the heading rules so a
    # single bad weight/minimum fails fast with one clear error.
    dimension_weights = _normalize_dimension_weights(raw.get("dimension_weights"), source)
    dimension_minimums = _normalize_dimension_minimums(raw.get("dimension_minimums"), source)

    expected_headers: list[ExpectedHeader] = []
    seen_headings: dict[str, str] = {}
    for index, rule in enumerate(raw["expected_headers"]):
        location = f"expected_headers[{index}]"
        if not isinstance(rule, dict):
            raise QualityConfigError(f"{source}: {location} must be a mapping")
        _reject_unknown_keys(rule, _EXPECTED_HEADER_KEYS, source, location)

        heading = _normalize_heading(rule.get("heading"), source, f"{location}.heading")
        raw_alternatives = rule.get("alternatives", [])
        if not isinstance(raw_alternatives, list):
            raise QualityConfigError(f"{source}: {location}.alternatives must be a list")
        alternatives = [
            _normalize_heading(value, source, f"{location}.alternatives[{alternative_index}]")
            for alternative_index, value in enumerate(raw_alternatives)
        ]
        points = rule.get("points", 5)
        if not _is_finite_number(points) or points < 0:
            raise QualityConfigError(f"{source}: {location}.points must be a non-negative number")
        # message and suggestion are optional; omitted ones get generic text naming the headings.
        accepted_display = " or ".join(f"'{accepted}'" for accepted in (heading, *alternatives))
        message = rule.get("message", f"Missing expected heading: {accepted_display}")
        suggestion = rule.get(
            "suggestion",
            f"Add a {accepted_display} section"
            if not alternatives
            else f"Add one of these sections: {accepted_display}",
        )
        for field, value in (("message", message), ("suggestion", suggestion)):
            if not isinstance(value, str) or not value.strip():
                raise QualityConfigError(f"{source}: {location}.{field} must be a non-empty string when set")

        for accepted in (heading, *alternatives):
            if accepted in seen_headings:
                raise QualityConfigError(
                    f"{source}: duplicate heading '{accepted}' in {location} (already listed in {seen_headings[accepted]})"
                )
            seen_headings[accepted] = location

        expected_headers.append(
            {
                "heading": heading,
                "alternatives": alternatives,
                "points": points,
                "message": message,
                "suggestion": suggestion,
            }
        )

    # One immutable, normalized object: a folder run parses the file once and
    # every skill scores against this same frozen snapshot.
    config = {
        "expected_headers": expected_headers,
        "dimension_weights": dimension_weights,
        "dimension_minimums": dimension_minimums,
    }
    return cast("QualityConfig", _freeze(config))


def quality_config_digest(config: QualityConfig) -> str:
    """Return a stable digest of the normalized rules, independent of file formatting."""
    canonical = json.dumps(_to_plain(config), sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def quality_config_provenance(path: Path | None, config: QualityConfig) -> QualityConfigProvenance:
    """Describe the effective config. Only the file name is kept, so reports do not leak local paths."""
    return {
        "source": "custom" if path else "packaged",
        "file": path.name if path else None,
        "digest": quality_config_digest(config),
    }


def format_dimension_weights(weights: Mapping[str, object]) -> str:
    """Render weights in canonical order, e.g. 'correctness 35%, discoverability 25%'."""
    return ", ".join(
        f"{name} {format(float(weights[name]) * 100, 'g')}%" for name in QUALITY_DIMENSIONS if name in weights
    )


def format_dimension_minimums(minimums: Mapping[str, object]) -> str:
    """Render minimums in canonical order, e.g. 'reliability >= 60, efficiency >= 50'."""
    return ", ".join(
        f"{name} >= {format(float(minimums[name]), '.12g')}" for name in QUALITY_DIMENSIONS if name in minimums
    )
