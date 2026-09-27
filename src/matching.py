"""Interpretable feature generation and scoring for blocked entity pairs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Mapping

from normalize import (
    accent_folded_text,
    alphanumeric_form,
    content_tokens,
    number_signature,
    token_form,
    transliterated_text,
)


FEATURE_NAMES = (
    "name_exact",
    "name_folded_exact",
    "name_token_similarity",
    "name_token_jaccard",
    "name_char_similarity",
    "name_prefix_similarity",
    "name_translit_similarity",
    "address_exact",
    "address_token_similarity",
    "address_token_jaccard",
    "address_char_similarity",
    "number_overlap",
    "address_alnum_similarity",
    "country_match",
)


DEFAULT_WEIGHTS = {
    "name_exact": 0.24,
    "name_folded_exact": 0.10,
    "name_token_similarity": 0.10,
    "name_token_jaccard": 0.08,
    "name_char_similarity": 0.10,
    "name_prefix_similarity": 0.04,
    "name_translit_similarity": 0.08,
    "address_exact": 0.08,
    "address_token_similarity": 0.05,
    "address_token_jaccard": 0.04,
    "address_char_similarity": 0.03,
    "number_overlap": 0.03,
    "address_alnum_similarity": 0.01,
    "country_match": 0.02,
}


def _text(row: Mapping, field: str) -> str:
    value = row.get(field, "")
    return "" if value is None else str(value)


def _token_set(value: str, content: bool = False) -> set[str]:
    tokens = content_tokens(value) if content else token_form(value)
    return {token for token in tokens if len(token) >= 3}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _char_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _prefix_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right), 12)
    return sum(a == b for a, b in zip(left[:length], right[:length])) / length


def _number_set(value: str) -> set[str]:
    return set(number_signature(value).split())


def extract_features(s1_row: Mapping, candidate_row: Mapping) -> dict[str, float]:
    """Return component features for one already-blocked candidate pair."""
    s1_name = _text(s1_row, "business_name")
    candidate_name = _text(candidate_row, "business_name")
    s1_address = _text(s1_row, "business_address")
    candidate_address = _text(candidate_row, "business_address")

    name_left = _token_set(s1_name)
    name_right = _token_set(candidate_name)
    address_left = _token_set(s1_address)
    address_right = _token_set(candidate_address)
    name_norm_left = _text(s1_row, "name_norm") or _text(s1_row, "business_name")
    name_norm_right = _text(candidate_row, "name_norm") or _text(candidate_row, "business_name")
    address_norm_left = _text(s1_row, "address_norm") or _text(s1_row, "business_address")
    address_norm_right = _text(candidate_row, "address_norm") or _text(candidate_row, "business_address")

    name_norm_left = name_norm_left.casefold()
    name_norm_right = name_norm_right.casefold()
    address_norm_left = address_norm_left.casefold()
    address_norm_right = address_norm_right.casefold()
    folded_left = accent_folded_text(s1_name)
    folded_right = accent_folded_text(candidate_name)
    translit_left = transliterated_text(s1_name)
    translit_right = transliterated_text(candidate_name)
    alnum_left = alphanumeric_form(s1_address)
    alnum_right = alphanumeric_form(candidate_address)
    numbers_left = _number_set(s1_address)
    numbers_right = _number_set(candidate_address)

    return {
        "name_exact": float(bool(name_norm_left and name_norm_left == name_norm_right)),
        "name_folded_exact": float(bool(folded_left and folded_left == folded_right)),
        "name_token_similarity": _overlap(name_left, name_right),
        "name_token_jaccard": _jaccard(name_left, name_right),
        "name_char_similarity": _char_similarity(name_norm_left, name_norm_right),
        "name_prefix_similarity": _prefix_similarity(
            alphanumeric_form(s1_name), alphanumeric_form(candidate_name)
        ),
        "name_translit_similarity": _char_similarity(translit_left, translit_right),
        "address_exact": float(bool(address_norm_left and address_norm_left == address_norm_right)),
        "address_token_similarity": _overlap(address_left, address_right),
        "address_token_jaccard": _jaccard(address_left, address_right),
        "address_char_similarity": _char_similarity(address_norm_left, address_norm_right),
        "number_overlap": _overlap(numbers_left, numbers_right),
        "address_alnum_similarity": _char_similarity(alnum_left, alnum_right),
        "country_match": float(_text(s1_row, "country") == _text(candidate_row, "country")),
    }


def combine_features(features: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    """Combine component features into a bounded, interpretable score."""
    weights = weights or DEFAULT_WEIGHTS
    total = sum(weights.get(name, 0.0) for name in FEATURE_NAMES)
    if not total:
        return 0.0
    return sum(float(features.get(name, 0.0)) * weights.get(name, 0.0) for name in FEATURE_NAMES) / total


def score_candidate(
    s1_row: Mapping,
    candidate_row: Mapping,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return all component scores plus ``final_score`` for one pair."""
    features = extract_features(s1_row, candidate_row)
    features["final_score"] = combine_features(features, weights)
    return features
