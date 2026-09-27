"""Small, disk-backed training experiment on the existing blocking output."""

from __future__ import annotations
import csv
from pathlib import Path
import argparse

import polars as pl

from blocking import generate_candidates, load_source
from matching import FEATURE_NAMES, score_candidate
from train_matcher import choose_threshold, evaluate_thresholds, fit_weights


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"
RESULT_PATH = ROOT / "experiments" / "matching_experiment_results.tsv"


def write_submission_files(
    s1_ids: list[str],
    candidates: pl.DataFrame,
    scored_rows: list[dict],
    threshold: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write one row per S1 and guarantee matches are a candidate subset."""
    candidate_lists = (
        candidates.group_by("source1_entity_id")
        .agg(pl.col("candidate_entity_id").unique().sort().alias("candidate_entity_ids"))
    )
    scored = pl.DataFrame(scored_rows)
    matched = (
        scored.filter(pl.col("final_score") >= threshold)
        .group_by("source1_entity_id")
        .agg(pl.col("candidate_entity_id").unique().sort().alias("matched_entity_ids"))
    )
    rows = pl.DataFrame({"source1_entity_id": s1_ids}).join(candidate_lists, on="source1_entity_id", how="left").join(matched, on="source1_entity_id", how="left")
    rows = rows.with_columns([
        pl.when(pl.col("candidate_entity_ids").is_null())
        .then(pl.lit(""))
        .otherwise(pl.col("candidate_entity_ids").list.join(","))
        .alias("candidate_entity_ids"),
        pl.when(pl.col("matched_entity_ids").is_null())
        .then(pl.lit(""))
        .otherwise(pl.col("matched_entity_ids").list.join(","))
        .alias("matched_entity_ids"),
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    rows.select(["source1_entity_id", "matched_entity_ids"]).write_csv(
        matching_path, separator="\t", quote_style="never"
    )
    rows.select(["source1_entity_id", "candidate_entity_ids"]).write_csv(
        candidate_path, separator="\t", quote_style="never"
    )
    return matching_path, candidate_path


def load_ground_truth(path: Path, s1_ids: set[str]) -> pl.DataFrame:
    gt = pl.read_csv(path, separator="\t", infer_schema_length=1000)
    return (
        gt.filter(pl.col("source1_entity_id").is_in(s1_ids))
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
        .explode("matched_list")
        .rename({"matched_list": "candidate_entity_id"})
        .select(["source1_entity_id", "candidate_entity_id"])
    )


def score_candidates(candidates: pl.DataFrame, s1: pl.DataFrame, target_path: Path) -> list[dict]:
    s1_rows = {row["entity_id"]: row for row in s1.iter_rows(named=True)}
    candidate_ids = candidates["candidate_entity_id"].unique().to_list()
    target = (
        pl.scan_csv(target_path, separator="\t", infer_schema_length=1000)
        .filter(pl.col("entity_id").is_in(candidate_ids))
        .collect()
    )
    target_rows = {row["entity_id"]: row for row in target.iter_rows(named=True)}
    scored = []
    for row in candidates.iter_rows(named=True):
        left = s1_rows.get(row["source1_entity_id"])
        right = target_rows.get(row["candidate_entity_id"])
        if left is None or right is None:
            continue
        features = score_candidate(left, right)
        features.update({
            "source1_entity_id": row["source1_entity_id"],
            "candidate_entity_id": row["candidate_entity_id"],
        })
        scored.append(features)
    return scored


def bounded_target(path: Path, row_limit: int | None) -> tuple[Path, Path | None]:
    if row_limit is None:
        return path, None
    sample_path = ROOT / "experiments" / f"_matching_target_{path.stem}.tsv"
    (
        pl.scan_csv(path, separator="\t", infer_schema_length=1000)
        .head(row_limit)
        .collect()
        .write_csv(sample_path, separator="\t")
    )
    return sample_path, sample_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-sample", type=int, default=1000)
    parser.add_argument("--target-rows", type=int, default=None)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=100_000,
        help="Maximum candidate pairs retained per target source for this experiment.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()

    s1 = load_source(TRAIN / "train_source1.tsv")
    s1 = s1.sample(n=min(args.s1_sample, s1.height), seed=args.seed)
    scored = []
    candidate_frames = []
    for source_name in ("train_source2.tsv", "train_source3.tsv"):
        target_path, temporary_path = bounded_target(TRAIN / source_name, args.target_rows)
        candidates = (
            generate_candidates(s1, target_path)
            .head(args.max_candidates)
            .collect()
        )
        candidate_frames.append(candidates)
        print(f"{source_name} candidate pairs: {candidates.height:,}")
        scored.extend(score_candidates(candidates, s1, target_path))
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if not scored:
        raise RuntimeError("No candidate rows could be scored")

    positives = load_ground_truth(
        TRAIN / "train_ground_truth.tsv",
        set(s1["entity_id"].to_list()),
    )
    positive_set = {
        (row["source1_entity_id"], row["candidate_entity_id"])
        for row in positives.iter_rows(named=True)
    }
    for row in scored:
        row["label"] = int((row["source1_entity_id"], row["candidate_entity_id"]) in positive_set)

    # Split by S1 entity to avoid putting near-duplicate candidates in both sets.
    s1_ids = sorted({row["source1_entity_id"] for row in scored})
    split_at = max(1, int(len(s1_ids) * 0.8))
    train_ids = set(s1_ids[:split_at])
    train_rows = [row for row in scored if row["source1_entity_id"] in train_ids]
    validation_rows = [row for row in scored if row["source1_entity_id"] not in train_ids]
    weights = fit_weights(train_rows)
    results = evaluate_thresholds(validation_rows or train_rows, weights)
    selected = choose_threshold(results)

    matching_path, candidate_path = write_submission_files(
        s1["entity_id"].to_list(),
        pl.concat(candidate_frames) if candidate_frames else pl.DataFrame({
            "source1_entity_id": [], "candidate_entity_id": []
        }),
        scored,
        float(selected["threshold"]),
        args.output.parent / "matching_outputs",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print("Learned weights:")
    for name in FEATURE_NAMES:
        print(f"  {name}: {weights[name]:.4f}")
    print("Threshold evaluation:")
    print(pl.DataFrame(results))
    print(f"Selected threshold: {selected['threshold']:.2f}")
    print(f"Saved candidate pairs to: {candidate_path}")
    print(f"Saved matching results to: {matching_path}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
