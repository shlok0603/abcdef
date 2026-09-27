from pathlib import Path
from collections import defaultdict

import polars as pl

from normalize import (
    normalize_text,
    accent_folded_text,
    alphanumeric_form,
    token_form,
    number_signature,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

# IMPORTANT for your 16 GB machine.
BATCH_SIZE = 50_000

# A block containing more records than this is ignored.
MAX_BLOCK_SIZE = 300

# Number of S1 entities used for this experiment.
SAMPLE_SIZE = 200


# ============================================================
# NORMALIZATION
# ============================================================

def norm(value):
    if value is None:
        return ""
    return normalize_text(value)


def folded(value):
    if value is None:
        return ""
    return accent_folded_text(value)


def alnum(value):
    if value is None:
        return ""
    return alphanumeric_form(value)


def tokens(value):
    if not value:
        return []

    return [
        x for x in token_form(value)
        if len(x) >= 3
    ]


def numbers(value):
    if value is None:
        return ""

    return number_signature(value)


# ============================================================
# BLOCKING KEYS
# ============================================================

def make_keys(country, name, address):

    name_norm = norm(name)
    name_folded = folded(name)
    name_alnum = alnum(name)

    address_norm = norm(address)

    name_tokens = tokens(name_norm)
    address_tokens = tokens(address_norm)

    number_sig = numbers(address_norm)

    keys = []

    # --------------------------------------------------------
    # 1. Exact normalized name
    # --------------------------------------------------------

    if name_norm:
        keys.append(
            (
                "NAME_EXACT",
                f"{country}|{name_norm}",
            )
        )

    # --------------------------------------------------------
    # 2. Accent-folded name
    # --------------------------------------------------------

    if name_folded:
        keys.append(
            (
                "NAME_FOLDED",
                f"{country}|{name_folded}",
            )
        )

    # --------------------------------------------------------
    # 3. Two meaningful name tokens
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        pair = tuple(sorted(name_tokens[:2]))

        keys.append(
            (
                "NAME_TOKEN2",
                f"{country}|{pair[0]}|{pair[1]}",
            )
        )

    # --------------------------------------------------------
    # 4. Six-character name prefix
    # --------------------------------------------------------

    if len(name_alnum) >= 6:

        keys.append(
            (
                "NAME_PREFIX6",
                f"{country}|{name_alnum[:6]}",
            )
        )

    # --------------------------------------------------------
    # 5. First address number
    # --------------------------------------------------------

    if number_sig:

        first_number = number_sig.split()[0]

        keys.append(
            (
                "ADDRESS_NUMBER",
                f"{country}|{first_number}",
            )
        )

    # --------------------------------------------------------
    # 6. Name prefix + address number
    # --------------------------------------------------------

    if name_alnum and number_sig:

        first_number = number_sig.split()[0]

        keys.append(
            (
                "NAME_NUMBER",
                f"{country}|{name_alnum[:6]}|{first_number}",
            )
        )

    # --------------------------------------------------------
    # 7. Two address tokens
    # --------------------------------------------------------

    if len(address_tokens) >= 2:

        pair = tuple(sorted(address_tokens[:2]))

        keys.append(
            (
                "ADDRESS_TOKEN2",
                f"{country}|{pair[0]}|{pair[1]}",
            )
        )

    return keys


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth_sample():

    print("Loading ground truth...")

    gt = pl.read_csv(
        GT_FILE,
        separator="\t",
        infer_schema_length=1000,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    gt = gt.filter(
        pl.col("matched_entity_ids").is_not_null()
        & (
            pl.col("matched_entity_ids")
            .str.strip_chars()
            != ""
        )
    )

    gt = gt.sample(
        n=min(SAMPLE_SIZE, gt.height),
        seed=42,
    )

    return gt


# ============================================================
# S1 LOADING
# ============================================================

def load_s1_sample(gt):

    ids = gt["source1_entity_id"].to_list()

    print("Loading sampled S1 records...")

    s1 = (
        pl.scan_csv(
            S1_FILE,
            separator="\t",
            infer_schema_length=1000,
        )
        .filter(
            pl.col("entity_id").is_in(ids)
        )
        .select([
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ])
        .collect()
    )

    return s1


# ============================================================
# BUILD SAMPLE BLOCKS
# ============================================================

def build_sample_block_keys(s1):

    print("Building blocking keys for sampled S1...")

    sample_keys = defaultdict(set)

    for row in s1.iter_rows(named=True):

        keys = make_keys(
            row["country"],
            row["business_name"],
            row["business_address"],
        )

        for key in keys:
            sample_keys[key].add(
                row["entity_id"]
            )

    return sample_keys


# ============================================================
# STREAM S2
# ============================================================

def stream_s2(sample_keys):

    print("\nStarting streamed S2 scan...")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Maximum block size: {MAX_BLOCK_SIZE:,}")

    # Candidate IDs for each S1.
    candidates = defaultdict(set)

    # Number of S2 records scanned.
    rows_scanned = 0

    # Statistics for each block.
    block_hits = defaultdict(int)

    batch_number = 0

    reader = pl.read_csv_batched(
        S2_FILE,
        separator="\t",
        infer_schema_length=1000,
        batch_size=BATCH_SIZE,
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    while True:

        batches = reader.next_batches(1)

        if not batches:
            break

        batch = batches[0]

        batch_number += 1
        rows_scanned += batch.height

        print(
            f"\rProcessed S2 rows: "
            f"{rows_scanned:,}",
            end="",
            flush=True,
        )

        for row in batch.iter_rows(named=True):

            keys = make_keys(
                row["country"],
                row["business_name"],
                row["business_address"],
            )

            for key in keys:

                # Only process blocks that correspond
                # to one of our sampled S1 records.
                if key not in sample_keys:
                    continue

                block_hits[key] += 1

                # We only keep the candidate if the block
                # has not exceeded our limit.
                if block_hits[key] <= MAX_BLOCK_SIZE:

                    for s1_id in sample_keys[key]:

                        candidates[s1_id].add(
                            row["entity_id"]
                        )

    print()
    print(f"Finished S2 scan.")
    print(f"Batches processed: {batch_number:,}")
    print(f"Rows scanned: {rows_scanned:,}")

    return candidates, block_hits


# ============================================================
# GROUND-TRUTH EVALUATION
# ============================================================

def evaluate(gt, candidates):

    total_positive = 0
    recovered_positive = 0

    candidate_counts = []

    recall_by_s1 = []

    for row in gt.iter_rows(named=True):

        s1_id = row["source1_entity_id"]

        true_ids = {
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip().startswith("S2-")
        }

        candidate_ids = candidates.get(
            s1_id,
            set(),
        )

        total_positive += len(true_ids)

        recovered = true_ids.intersection(
            candidate_ids
        )

        recovered_positive += len(recovered)

        candidate_counts.append(
            len(candidate_ids)
        )

        if true_ids:
            recall_by_s1.append(
                len(recovered) / len(true_ids)
            )

    recall = (
        recovered_positive / total_positive
        if total_positive
        else 0
    )

    candidate_counts.sort()

    mean_candidates = (
        sum(candidate_counts) / len(candidate_counts)
        if candidate_counts
        else 0
    )

    median_candidates = (
        candidate_counts[len(candidate_counts) // 2]
        if candidate_counts
        else 0
    )

    max_candidates = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )

    zero_candidate = sum(
        x == 0 for x in candidate_counts
    )

    full_recall = sum(
        x == 1 for x in recall_by_s1
    )

    return {
        "total_positive": total_positive,
        "recovered_positive": recovered_positive,
        "recall": recall,
        "mean_candidates": mean_candidates,
        "median_candidates": median_candidates,
        "max_candidates": max_candidates,
        "zero_candidate_s1": zero_candidate,
        "full_recall_entities": full_recall,
        "sample_entities": len(candidate_counts),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MEMORY-SAFE STREAMING BLOCKING EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt = load_ground_truth_sample()

    print(
        f"Sampled positive S1 entities: "
        f"{gt.height:,}"
    )

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    s1 = load_s1_sample(gt)

    print(
        f"S1 records loaded: "
        f"{s1.height:,}"
    )

    # --------------------------------------------------------
    # Build only the S1-side keys
    # --------------------------------------------------------

    sample_keys = build_sample_block_keys(s1)

    print(
        f"S1 blocking keys: "
        f"{len(sample_keys):,}"
    )

    # --------------------------------------------------------
    # Stream S2
    # --------------------------------------------------------

    candidates, block_hits = stream_s2(
        sample_keys
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate(
        gt,
        candidates,
    )

    print("\n")
    print("=" * 80)
    print("BLOCKING RESULTS")
    print("=" * 80)

    print(
        f"\nGround-truth S2 positive pairs:"
        f" {results['total_positive']:,}"
    )

    print(
        f"Recovered by blocker:"
        f" {results['recovered_positive']:,}"
    )

    print(
        f"\nBLOCKING RECALL:"
        f" {results['recall']:.4%}"
    )

    print(
        f"\nAverage candidates/S1:"
        f" {results['mean_candidates']:.2f}"
    )

    print(
        f"Median candidates/S1:"
        f" {results['median_candidates']:,}"
    )

    print(
        f"Maximum candidates/S1:"
        f" {results['max_candidates']:,}"
    )

    print(
        f"S1 entities with ZERO candidates:"
        f" {results['zero_candidate_s1']:,}"
    )

    print(
        f"S1 entities with 100% positive recall:"
        f" {results['full_recall_entities']:,}"
        f" / {results['sample_entities']:,}"
    )

    print("\n")
    print("=" * 80)
    print("STREAMING EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()