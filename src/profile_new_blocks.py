from pathlib import Path
from collections import defaultdict
import polars as pl
import numpy as np

from normalize import (
    normalize_text,
    accent_folded_text,
    alphanumeric_form,
    token_form,
    number_signature,
)

ROOT = Path(__file__).resolve().parents[1]

S1_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_source1.tsv"
S2_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_source2.tsv"

BATCH_SIZE = 50_000
S1_SAMPLE_SIZE = 2000
SEED = 42


def make_keys(row):

    country = str(row.get("country") or "")

    name = normalize_text(
        row.get("business_name") or ""
    )

    folded = accent_folded_text(
        row.get("business_name") or ""
    )

    address = normalize_text(
        row.get("business_address") or ""
    )

    name_alnum = alphanumeric_form(
        row.get("business_name") or ""
    )

    address_alnum = alphanumeric_form(
        row.get("business_address") or ""
    )

    numbers = number_signature(
        row.get("business_address") or ""
    )

    name_tokens = [
        t for t in token_form(name)
        if len(t) >= 3
    ]

    address_tokens = [
        t for t in token_form(address)
        if len(t) >= 3
    ]

    keys = set()

    if name:
        keys.add(("N_EXACT", f"N_EXACT|{country}|{name}"))

    if folded:
        keys.add(("N_FOLDED", f"N_FOLDED|{country}|{folded}"))

    for token in name_tokens:
        keys.add(("N_TOKEN", f"N_TOKEN|{country}|{token}"))

    if len(name_tokens) >= 2:

        keys.add((
            "N_PAIR",
            f"N_PAIR|{country}|"
            f"{name_tokens[0]}|{name_tokens[1]}"
        ))

        pair = sorted(
            [name_tokens[0], name_tokens[1]]
        )

        keys.add((
            "N_PAIR_SORTED",
            f"N_PAIR_SORTED|{country}|"
            f"{pair[0]}|{pair[1]}"
        ))

    if len(name_tokens) >= 2:

        sorted_tokens = sorted(
            set(name_tokens)
        )

        keys.add((
            "N_SET",
            f"N_SET|{country}|"
            f"{'|'.join(sorted_tokens[:4])}"
        ))

    if len(name_alnum) >= 5:
        keys.add((
            "N_PREFIX",
            f"N_PREFIX|{country}|"
            f"{name_alnum[:5]}"
        ))

    for token in address_tokens:
        keys.add((
            "A_TOKEN",
            f"A_TOKEN|{country}|{token}"
        ))

    if len(address_tokens) >= 2:

        pair = sorted(
            [
                address_tokens[0],
                address_tokens[1],
            ]
        )

        keys.add((
            "A_PAIR",
            f"A_PAIR|{country}|"
            f"{pair[0]}|{pair[1]}"
        ))

    if len(address_tokens) >= 2:

        sorted_address = sorted(
            set(address_tokens)
        )

        keys.add((
            "A_SET",
            f"A_SET|{country}|"
            f"{'|'.join(sorted_address[:4])}"
        ))

    if numbers:

        first_number = numbers.split()[0]

        keys.add((
            "A_NUMBER",
            f"A_NUMBER|{country}|{first_number}"
        ))

        if address_tokens:

            keys.add((
                "A_NUM_TOKEN",
                f"A_NUM_TOKEN|{country}|"
                f"{first_number}|{address_tokens[0]}"
            ))

    if name_alnum and numbers:

        first_number = numbers.split()[0]

        keys.add((
            "NAME_NUMBER",
            f"NAME_NUMBER|{country}|"
            f"{name_alnum[:5]}|{first_number}"
        ))

    if name_tokens and numbers:

        first_number = numbers.split()[0]

        for token in name_tokens[:3]:

            keys.add((
                "TOKEN_NUMBER",
                f"TOKEN_NUMBER|{country}|"
                f"{token}|{first_number}"
            ))

    if address_alnum:

        keys.add((
            "A_ALNUM_PREFIX",
            f"A_ALNUM_PREFIX|{country}|"
            f"{address_alnum[:12]}"
        ))

    return keys


def main():

    print("=" * 80)
    print("NEW BLOCK PROFILE")
    print("=" * 80)

    print("\nLoading S1 sample...")

    s1 = pl.read_csv(
        S1_PATH,
        separator="\t",
        infer_schema_length=10000,
    ).sample(
        n=S1_SAMPLE_SIZE,
        seed=SEED,
    )

    s1_key_set = set()

    for row in s1.iter_rows(named=True):

        for key_type, key in make_keys(row):

            s1_key_set.add(
                (key_type, key)
            )

    print(
        f"S1 sample: {s1.height:,}"
    )

    print(
        f"S1 unique keys: "
        f"{len(s1_key_set):,}"
    )

    print("\nScanning S2...")

    frequencies = defaultdict(int)

    processed = 0

    reader = pl.read_csv_batched(
        S2_PATH,
        separator="\t",
        batch_size=BATCH_SIZE,
        infer_schema_length=10000,
    )

    while True:

        batches = reader.next_batches(1)

        if not batches:
            break

        batch = batches[0]

        for row in batch.iter_rows(named=True):

            for key_type, key in make_keys(row):

                full_key = (
                    key_type,
                    key,
                )

                if full_key in s1_key_set:
                    frequencies[full_key] += 1

        processed += batch.height

        if processed % 500_000 < BATCH_SIZE:

            print(
                f"Processed: {processed:,}"
            )

    print(
        f"\nFinished: {processed:,}"
    )

    print("\n" + "=" * 80)
    print("BLOCK FREQUENCY PROFILE")
    print("=" * 80)

    by_type = defaultdict(list)

    for (key_type, key), count in frequencies.items():

        by_type[key_type].append(count)

    header = (
        f"{'KEY TYPE':<20}"
        f"{'N':>8}"
        f"{'MEAN':>12}"
        f"{'MEDIAN':>12}"
        f"{'P90':>12}"
        f"{'P95':>12}"
        f"{'P99':>12}"
        f"{'MAX':>12}"
    )

    print(header)
    print("-" * len(header))

    for key_type in sorted(by_type):

        values = np.array(
            by_type[key_type],
            dtype=np.int64,
        )

        print(
            f"{key_type:<20}"
            f"{len(values):>8,}"
            f"{values.mean():>12.2f}"
            f"{np.median(values):>12.0f}"
            f"{np.percentile(values, 90):>12.0f}"
            f"{np.percentile(values, 95):>12.0f}"
            f"{np.percentile(values, 99):>12.0f}"
            f"{values.max():>12,}"
        )


if __name__ == "__main__":
    main()