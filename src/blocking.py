from pathlib import Path
import gc
import polars as pl
import re

from normalize import (
    normalize_text,
    accent_folded_text,
    alphanumeric_form,
    token_form,
    number_signature,
    content_token_form,
    transliterated_text,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MIN_NAME_TOKEN_LENGTH = 3
MAX_BLOCK_SIZE = 5000


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_source(path):
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
    )


# ------------------------------------------------------------
# Normalized columns
# ------------------------------------------------------------

def add_normalized_columns(df: pl.DataFrame) -> pl.DataFrame:

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

        pl.col("business_address")
        .map_elements(
            normalize_text,
            return_dtype=pl.String,
        )
        .alias("address_norm"),

        pl.col("business_address")
        .map_elements(
            accent_folded_text,
            return_dtype=pl.String,
        )
        .alias("address_folded"),

        pl.col("business_name")
        .map_elements(
            alphanumeric_form,
            return_dtype=pl.String,
        )
        .alias("name_alnum"),

        pl.col("business_address")
        .map_elements(
            alphanumeric_form,
            return_dtype=pl.String,
        )
        .alias("address_alnum"),

        pl.col("business_name")
        .map_elements(
            content_token_form,
            return_dtype=pl.String,
        )
        .alias("name_content"),

        pl.col("business_address")
        .map_elements(
            number_signature,
            return_dtype=pl.String,
        )
        .alias("address_numbers"),
    ])


# ------------------------------------------------------------
# Blocking-key helpers
# ------------------------------------------------------------

def first_tokens(text, n=2):

    tokens = token_form(text)

    tokens = [
        t for t in tokens
        if len(t) >= MIN_NAME_TOKEN_LENGTH
    ]

    return tokens[:n]


def create_blocking_keys(df: pl.DataFrame) -> pl.DataFrame:

    records = df.select([
        "entity_id",
        "country",
        "name_norm",
        "name_folded",
        "address_norm",
        "address_folded",
        "name_alnum",
        "address_alnum",
        "name_content",
        "address_numbers",
    ]).iter_rows(named=True)

    output = []

    for row in records:

        country = row["country"]

        name = row["name_norm"] or ""
        folded = row["name_folded"] or ""
        address = row["address_norm"] or ""

        name_alnum = row["name_alnum"] or ""
        address_alnum = row["address_alnum"] or ""

        numbers = row["address_numbers"] or ""

        # ----------------------------------------------------
        # TRANSLITERATION
        # ----------------------------------------------------

        name_translit = transliterated_text(name)
        name_translit_alnum = alphanumeric_form(name_translit)

        translit_tokens = [
            t for t in token_form(name_translit)
            if len(t) >= MIN_NAME_TOKEN_LENGTH
        ]

        name_tokens = [
            t for t in token_form(name)
            if len(t) >= MIN_NAME_TOKEN_LENGTH
        ]

        address_tokens = [
            t for t in token_form(address)
            if len(t) >= 3
        ]

        keys = set()

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        if name:
            keys.add(
                f"N_EXACT|{country}|{name}"
            )

        if folded:
            keys.add(
                f"N_FOLDED|{country}|{folded}"
            )

        for token in name_tokens:
            keys.add(
                f"N_TOKEN|{country}|{token}"
            )

        if len(name_tokens) >= 2:

            keys.add(
                f"N_PAIR|{country}|"
                f"{name_tokens[0]}|{name_tokens[1]}"
            )

            pair = sorted([
                name_tokens[0],
                name_tokens[1],
            ])

            keys.add(
                f"N_PAIR_SORTED|{country}|"
                f"{pair[0]}|{pair[1]}"
            )

        if len(name_tokens) >= 2:

            sorted_tokens = sorted(set(name_tokens))

            keys.add(
                f"N_SET|{country}|"
                f"{'|'.join(sorted_tokens[:4])}"
            )

        if len(name_alnum) >= 5:

            keys.add(
                f"N_PREFIX|{country}|"
                f"{name_alnum[:5]}"
            )

        # ----------------------------------------------------
        # EXTRA NAME KEYS
        # ----------------------------------------------------

        if name_alnum:

            keys.add(
                f"N_SUFFIX|{country}|"
                f"{name_alnum[-8:]}"
            )

            keys.add(
                f"N_LENGTH_PREFIX|{country}|"
                f"{len(name_alnum)}:{name_alnum[:4]}"
            )

        # ----------------------------------------------------
        # TRANSLITERATED NAME KEYS
        # ----------------------------------------------------

        if name_translit:

            keys.add(
                f"N_TRANSLIT|{country}|"
                f"{name_translit}"
            )

        if len(name_translit_alnum) >= 5:

            keys.add(
                f"N_TRANSLIT_PREFIX|{country}|"
                f"{name_translit_alnum[:6]}"
            )

        for token in translit_tokens:

            keys.add(
                f"N_TRANSLIT_TOKEN|{country}|"
                f"{token}"
            )

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        for token in address_tokens:

            keys.add(
                f"A_TOKEN|{country}|{token}"
            )

        if len(address_tokens) >= 2:

            pair = sorted([
                address_tokens[0],
                address_tokens[1],
            ])

            keys.add(
                f"A_PAIR|{country}|"
                f"{pair[0]}|{pair[1]}"
            )

        if len(address_tokens) >= 2:

            sorted_address = sorted(set(address_tokens))

            keys.add(
                f"A_SET|{country}|"
                f"{'|'.join(sorted_address[:4])}"
            )

        if numbers:

            first_number = numbers.split()[0]

            keys.add(
                f"A_NUMBER|{country}|"
                f"{first_number}"
            )

            if address_tokens:

                keys.add(
                    f"A_NUM_TOKEN|{country}|"
                    f"{first_number}|"
                    f"{address_tokens[0]}"
                )

        # ----------------------------------------------------
        # NAME + ADDRESS
        # ----------------------------------------------------

        if name_alnum and numbers:

            first_number = numbers.split()[0]

            keys.add(
                f"NAME_NUMBER|{country}|"
                f"{name_alnum[:5]}|"
                f"{first_number}"
            )

        if name_tokens and numbers:

            first_number = numbers.split()[0]

            for token in name_tokens[:3]:

                keys.add(
                    f"TOKEN_NUMBER|{country}|"
                    f"{token}|{first_number}"
                )

        # ----------------------------------------------------
        # ADDRESS ALPHANUMERIC
        # ----------------------------------------------------

        if address_alnum:

            keys.add(
                f"A_ALNUM_PREFIX|{country}|"
                f"{address_alnum[:12]}"
            )

            keys.add(
                f"A_ALNUM_SUFFIX|{country}|"
                f"{address_alnum[-12:]}"
            )

            keys.add(
                f"COUNTRY_ADDRESS|{country}|"
                f"{address_alnum[:20]}"
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        for key in keys:

            output.append({
                "entity_id": row["entity_id"],
                "country": country,
                "block_key": key,
            })

    return pl.DataFrame(output)
# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------

def generate_candidates(
    s1: pl.DataFrame,
    target_path: Path,
):
    print("\nAdding normalized columns to S1...")

    s1 = add_normalized_columns(s1)

    print("Creating S1 blocking keys...")

    s1_blocks = create_blocking_keys(s1)
    del s1
    gc.collect()

    print(
        f"S1 block rows: {s1_blocks.height:,}"
    )

    print(
        f"S1 unique blocking keys: "
        f"{s1_blocks['block_key'].n_unique():,}"
    )

    output_dir = (
        ROOT
        / "experiments"
        / "candidate_parts"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old parts
    for old_file in output_dir.glob("part_*.tsv"):
        old_file.unlink()

    final_path = ROOT / "experiments" / "blocking_candidates.tsv"
    if final_path.exists():
        final_path.unlink()

    first = True
    total_candidates = 0
    max_batch_candidates = 0

    print(
        "\nStreaming target in 10,000-row batches..."
    )

    reader = pl.read_csv_batched(
        target_path,
        separator="\t",
        has_header=True,
        batch_size=10_000,
        infer_schema_length=1000,
    )

    processed = 0
    batches_written = 0

    while True:

        batches = reader.next_batches(1)

        if not batches:
            break

        target = batches[0]

        processed += target.height

        target = add_normalized_columns(target)

        target_blocks = create_blocking_keys(target)

        if target_blocks.height == 0:
            continue

        # Count target-side block frequency
        block_counts = (
            target_blocks
            .group_by("block_key")
            .agg(
                pl.len().alias("target_count")
            )
        )

        target_blocks = target_blocks.join(
            block_counts,
            on="block_key",
            how="left",
        )

        # Keep the existing 50,000-row block cap, then join this batch only
        # against the in-memory S1 block lookup. No Python candidate list is
        # created here.
        eligible_blocks = target_blocks.filter(
            pl.col("target_count") <= 50_000
        )

        part = (
            s1_blocks.select(["entity_id", "block_key"])
            .join(
                eligible_blocks.select(["block_key", "entity_id"]),
                on="block_key",
                how="inner",
                suffix="_target",
            )
            .select([
                pl.col("entity_id").alias("source1_entity_id"),
                pl.col("entity_id_target").alias("candidate_entity_id"),
            ])
            .unique()
        )

        batch_candidates = part.height
        total_candidates += batch_candidates
        max_batch_candidates = max(max_batch_candidates, batch_candidates)

        if batch_candidates:
            mode = "wb" if first else "ab"
            with final_path.open(mode) as handle:
                part.write_csv(
                    handle,
                    separator="\t",
                    include_header=first,
                )
            first = False
            batches_written += 1

        del target_blocks
        del block_counts
        del eligible_blocks
        del part
        del target
        gc.collect()

        if processed % 100_000 == 0 or processed <= 10_000:
            print(
                f"Processed target rows: {processed:,}; "
                f"candidates written: {total_candidates:,}"
            )

    print(
            f"\nCandidate batches written: "
            f"{batches_written:,}"
    )

    print(
        f"Total candidates written: {total_candidates:,}; "
        f"largest batch: {max_batch_candidates:,}"
    )

    print(
        f"Candidate directory:\n"
        f"{output_dir}"
    )

    if first:
        pl.DataFrame(
            schema={
                "source1_entity_id": pl.String,
                "candidate_entity_id": pl.String,
            }
        ).write_csv(final_path, separator="\t")

    print(
        f"\nSaved candidates to:\n"
        f"{final_path}"
    )

    return pl.scan_csv(
        final_path,
        separator="\t",
    )
# ------------------------------------------------------------
# Candidate statistics
# ------------------------------------------------------------

def candidate_statistics(candidates):

    counts = (
        candidates
        .group_by("source1_entity_id")
        .agg(
            pl.len().alias("candidate_count")
        )
    )

    print("\nCandidate statistics:")

    print(
        counts
        .select([
            pl.len().alias("S1 entities"),
            pl.col("candidate_count").mean().alias("mean"),
            pl.col("candidate_count").median().alias("median"),
            pl.col("candidate_count").max().alias("max"),
        ])
        .collect(engine="streaming")
    )

    print("\nCandidate count distribution:")

    print(
        counts
        .select("candidate_count")
        .describe()
        .collect(engine="streaming")
    )


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():

    print("=" * 80)
    print("BLOCKING EXPERIMENT")
    print("=" * 80)

    # --------------------------------------------------------
    # Load S1 only
    # S2 will be streamed in 50,000-row batches
    # --------------------------------------------------------

    print("\nLoading S1...")

    s1 = load_source(
        TRAIN / "train_source1.tsv"
    )

    # Small S1 experiment first
    s1_sample = s1.sample(
        n=min(10_000, s1.height),
        seed=42,
    )
    del s1
    gc.collect()

    print(
        f"\nS1 sample: "
        f"{s1_sample.height:,}"
    )

    # --------------------------------------------------------
    # Stream S2 and generate candidates
    # --------------------------------------------------------

    candidates = generate_candidates(
        s1_sample,
        TRAIN / "train_source2.tsv",
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    candidate_statistics(candidates)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        ROOT
        / "experiments"
        / "blocking_sample.tsv"
    )

    candidates.sink_csv(
        output_path,
        separator="\t",
    )

    print(
        f"\nSaved candidates to:\n"
        f"{output_path}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "BLOCKING EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )

if __name__ == "__main__":
    main()