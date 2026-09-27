"""Fit simple pair-score weights and evaluate precision-heavy thresholds."""

from __future__ import annotations

from typing import Iterable, Mapping

from matching import DEFAULT_WEIGHTS, FEATURE_NAMES, combine_features


THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def fit_weights(rows: Iterable[Mapping]) -> dict[str, float]:
    """Learn non-negative weights from positive/negative candidate rows.

    The weight is proportional to the positive-minus-negative mean, with the
    existing conservative defaults retained when a feature has no signal.
    """
    positive_count = 0
    negative_count = 0
    positive_sums = {name: 0.0 for name in FEATURE_NAMES}
    negative_sums = {name: 0.0 for name in FEATURE_NAMES}
    for row in rows:
        sums = positive_sums if int(row["label"]) else negative_sums
        if int(row["label"]):
            positive_count += 1
        else:
            negative_count += 1
        for name in FEATURE_NAMES:
            sums[name] += float(row.get(name, 0.0))
    if not positive_count or not negative_count:
        return dict(DEFAULT_WEIGHTS)

    raw = {}
    for name in FEATURE_NAMES:
        positive_mean = positive_sums[name] / positive_count
        negative_mean = negative_sums[name] / negative_count
        raw[name] = max(0.0, positive_mean - negative_mean)

    total = sum(raw.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: raw[name] / total for name in FEATURE_NAMES}


def evaluate_thresholds(rows: Iterable[Mapping], weights: Mapping[str, float]) -> list[dict[str, float]]:
    counts = {
        threshold: {"predicted_matches": 0, "true_positives": 0,
                    "false_positives": 0, "false_negatives": 0}
        for threshold in THRESHOLDS
    }
    total_positives = 0
    for row in rows:
        label = int(row["label"])
        total_positives += label
        score = combine_features(row, weights)
        for threshold in THRESHOLDS:
            predicted = score >= threshold
            if predicted:
                counts[threshold]["predicted_matches"] += 1
                counts[threshold]["true_positives"] += label
                counts[threshold]["false_positives"] += not label
            elif label:
                counts[threshold]["false_negatives"] += 1
    results = []
    for threshold in THRESHOLDS:
        counts_for_threshold = counts[threshold]
        true_positive = counts_for_threshold["true_positives"]
        false_positive = counts_for_threshold["false_positives"]
        false_negative = counts_for_threshold["false_negatives"]
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f05 = (1.25 * precision * recall / (0.25 * precision + recall)) if precision and recall else 0.0
        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f0_5": f05,
            **counts_for_threshold,
        })
    return results


def choose_threshold(results: Iterable[Mapping[str, float]]) -> Mapping[str, float]:
    return max(results, key=lambda row: (row["f0_5"], row["precision"], row["recall"]))
