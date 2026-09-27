from pathlib import Path
from collections import defaultdict, Counter

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


# Maximum number of target records allowed in a block.
MAX_BLOCK_SIZE = 300


def load_source(path):

    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
    )


def normalize_columns(df):

    return df.with_columns([
        pl.col("business_name")
        .map_elements(
            normalize_text,
            return_dtype=pl.String,
        )
        .alias("name_norm"),

        pl.col("business_name")
        .map_elements(
            accent_folded_text,
            return_dtype=pl.String,
        )
        .alias("name_folded"),

        pl.col("business_name")
        .map_elements(
            alphanumeric_form,
            return_dtype=pl.String,
        )
        .alias("name_alnum"),

        pl.col("business_address")
        .map_elements(
            normalize_text,
            return_dtype=pl.String,
        )
        .alias("address_norm"),

        pl.col("business_address")
        .map_elements(
            number_signature,
            return_dtype=pl.String,
        )
        .alias("address_numbers"),
    ])


def get_tokens(text):

    if not text:
        return []

    return [
        x for x in token_form(text)
        if len(x) >= 3
    ]


def make_keys(row):

    country = row["country"]

    name = row["name_norm"] or ""
    folded = row["name_folded"] or ""
    alnum = row["name_alnum"] or ""
    address = row["address_norm"] or ""
    numbers = row["address_numbers"] or ""

    name_tokens = get_tokens(name)
    address_tokens = get_tokens(address)

    keys = []

    # --------------------------------------------------------
    # 1. Exact normalized name
    # --------------------------------------------------------

    if name:
        keys.append(
            ("NAME_EXACT", f"{country}|{name}")
        )

    # --------------------------------------------------------
    # 2. Accent-folded exact name
    # --------------------------------------------------------

    if folded:
        keys.append(
            ("NAME_FOLDED", f"{country}|{folded}")
        )

    # --------------------------------------------------------
    # 3. Two-name-token block
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        # Use first two tokens sorted so token order doesn't matter.
        pair = tuple(sorted(name_tokens[:2]))

        keys.append(
            (
                "NAME_TOKEN2",
                f"{country}|{pair[0]}|{pair[1]}"
            )
        )

    # --------------------------------------------------------
    # 4. Name prefix
    # --------------------------------------------------------

    if len(alnum) >= 6:

        keys.append(
            (
                "NAME_PREFIX6",
                f"{country}|{alnum[:6]}"
            )
        )

    # --------------------------------------------------------
    # 5. Address number
    # --------------------------------------------------------

    if numbers:

        first_number = numbers.split()[0]

        keys.append(
            (
                "ADDRESS_NUMBER",
                f"{country}|{first_number}"
            )
        )

    # --------------------------------------------------------
    # 6. Name prefix + address number
    # --------------------------------------------------------

    if alnum and numbers:

        first_number = numbers.split()[0]

        keys.append(
            (
                "NAME_NUMBER",
                f"{country}|{alnum[:6]}|{first_number}"
            )
        )

    # --------------------------------------------------------
    # 7. Address token pair
    # --------------------------------------------------------

    if len(address_tokens) >= 2:

        pair = tuple(sorted(address_tokens[:2]))

        keys.append(
            (
                "ADDRESS_TOKEN2",
                f"{country}|{pair[0]}|{pair[1]}"
            )
        )

    return keys


def build_index(df):

    index = defaultdict(list)

    key_counts = Counter()

    rows = df.iter_rows(named=True)

    for row in rows:

        keys = make_keys(row)

        for key_type, key in keys:

            full_key = (key_type, key)

            index[full_key].append(
                row["entity_id"]
            )

            key_counts[full_key] += 1

    return index, key_counts


def filter_blocks(index, key_counts):

    filtered = {}

    removed = 0

    for key, ids in index.items():

        if key_counts[key] <= MAX_BLOCK_SIZE:

            filtered[key] = ids

        else:

            removed += 1

    return filtered, removed


def main():

    print("=" * 80)
    print("BLOCKING RECALL EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    print("\nLoading ground truth...")

    gt = pl.read_csv(
        GT_FILE,
        separator="\t",
        infer_schema_length=1000,
    )

    gt = gt.filter(
        pl.col("matched_entity_ids").is_not_null()
        & (
            pl.col("matched_entity_ids")
            .str.strip_chars()
            != ""
        )
    )

    # Sample S1 entities with known positives.
    gt_sample = gt.sample(
        n=min(200, gt.height),
        seed=42,
    )

    print(
        f"Sample S1 entities: "
        f"{gt_sample.height:,}"
    )

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print("\nLoading S1...")

    s1_ids = gt_sample["source1_entity_id"].to_list()

    s1 = (
        pl.scan_csv(
            S1_FILE,
            separator="\t",
            infer_schema_length=1000,
        )
        .filter(
            pl.col("entity_id")
            .is_in(s1_ids)
        )
        .collect()
    )

    # --------------------------------------------------------
    # Extract all S2 ground-truth IDs
    # --------------------------------------------------------

    pairs = (
        gt_sample
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("matched_list")
        )
        .explode("matched_list")
        .rename({
            "matched_list": "matched_entity_id"
        })
    )

    s2_positive_ids = (
        pairs
        .filter(
            pl.col("matched_entity_id")
            .str.starts_with("S2-")
        )
        ["matched_entity_id"]
        .to_list()
    )

    print(
        f"Ground-truth S2 positives: "
        f"{len(s2_positive_ids):,}"
    )

    # --------------------------------------------------------
    # Load all S2
    # --------------------------------------------------------

    print("\nLoading S2...")

    s2 = load_source(S2_FILE)

    print(
        f"S2 rows: {s2.height:,}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("\nNormalizing...")

    s1 = normalize_columns(s1)
    s2 = normalize_columns(s2)

    # --------------------------------------------------------
    # Build S2 block index
    # --------------------------------------------------------

    print("\nBuilding S2 block index...")

    s2_index, s2_counts = build_index(s2)

    print(
        f"Raw blocking keys: "
        f"{len(s2_index):,}"
    )

    filtered_index, removed = filter_blocks(
        s2_index,
        s2_counts,
    )

    print(
        f"Blocks removed because > "
        f"{MAX_BLOCK_SIZE} records: "
        f"{removed:,}"
    )

    print(
        f"Usable blocking keys: "
        f"{len(filtered_index):,}"
    )

    # --------------------------------------------------------
    # Evaluate each S1
    # --------------------------------------------------------

    recovered = 0
    total_positive = 0

    candidate_counts = []

    print("\nEvaluating blocking recall...")

    for s1_row in s1.iter_rows(named=True):

        s1_id = s1_row["entity_id"]

        # Ground truth IDs for this S1.
        gt_row = gt_sample.filter(
            pl.col("source1_entity_id")
            == s1_id
        )

        if gt_row.height == 0:
            continue

        gt_ids = (
            gt_row
            .select("matched_entity_ids")
            .item()
        )

        gt_ids = {
            x.strip()
            for x in gt_ids.split(",")
            if x.strip().startswith("S2-")
        }

        total_positive += len(gt_ids)

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        candidates = set()

        for key_type, key in make_keys(s1_row):

            full_key = (key_type, key)

            if full_key in filtered_index:

                candidates.update(
                    filtered_index[full_key]
                )

        candidate_counts.append(
            len(candidates)
        )

        recovered += len(
            gt_ids.intersection(candidates)
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    recall = (
        recovered / total_positive
        if total_positive
        else 0
    )

    print("\n")
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(
        f"\nGround-truth positive pairs: "
        f"{total_positive:,}"
    )

    print(
        f"Recovered positive pairs: "
        f"{recovered:,}"
    )

    print(
        f"BLOCKING RECALL: "
        f"{recall:.4%}"
    )

    if candidate_counts:

        print(
            f"\nAverage candidates/S1: "
            f"{sum(candidate_counts) / len(candidate_counts):.2f}"
        )

        print(
            f"Median candidates/S1: "
            f"{sorted(candidate_counts)[len(candidate_counts)//2]:,}"
        )

        print(
            f"Maximum candidates/S1: "
            f"{max(candidate_counts):,}"
        )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()