from pathlib import Path
from collections import defaultdict
import polars as pl
from normalize import transliterated_text
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
GT_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

BATCH_SIZE = 50_000
S1_SAMPLE_SIZE = 2000
SEED = 42

CAPS = {
    "N_EXACT": 250,
    "N_FOLDED": 250,
    "N_TOKEN": 20_000,
    "N_PAIR": 8_000,
    "N_PAIR_SORTED": 8_000,
    "N_SET": 2_000,
    "N_PREFIX": 12_000,
    "A_TOKEN": 25_000,
    "A_PAIR": 8_000,
    "A_SET": 2_000,
    "A_NUMBER": 12_000,
    "A_NUM_TOKEN": 8_000,
    "NAME_NUMBER": 2_000,
    "TOKEN_NUMBER": 8_000,
    "A_ALNUM_PREFIX": 3_000,
    "N_SUFFIX": 5_000,
    "N_LENGTH_PREFIX": 5_000,
    "A_ALNUM_SUFFIX": 5_000,
    "COUNTRY_ADDRESS": 5_000,
    "N_TRANSLIT": 500,
    "N_TRANSLIT_PREFIX": 5_000,
    "N_TRANSLIT_TOKEN": 5_000,
}


def clean_tokens(text):
    if not text:
        return []

    return [
        t for t in token_form(text)
        if len(t) >= 3
    ]

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
    name_translit = transliterated_text(
        row.get("business_name") or ""
    )

    name_translit_alnum = alphanumeric_form(
        name_translit
    )

    translit_tokens = [
        t for t in token_form(name_translit)
        if len(t) >= 3
    ]

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

    # ----------------------------------------------------
    # NAME
    # ----------------------------------------------------

    if name:
        keys.add((
            "N_EXACT",
            f"N_EXACT|{country}|{name}"
        ))

    if folded:
        keys.add((
            "N_FOLDED",
            f"N_FOLDED|{country}|{folded}"
        ))

    for token in name_tokens:
        keys.add((
            "N_TOKEN",
            f"N_TOKEN|{country}|{token}"
        ))

    if len(name_tokens) >= 2:

        keys.add((
            "N_PAIR",
            f"N_PAIR|{country}|"
            f"{name_tokens[0]}|{name_tokens[1]}"
        ))

        pair = sorted([
            name_tokens[0],
            name_tokens[1]
        ])

        keys.add((
            "N_PAIR_SORTED",
            f"N_PAIR_SORTED|{country}|"
            f"{pair[0]}|{pair[1]}"
        ))

    if len(name_tokens) >= 2:

        sorted_tokens = sorted(set(name_tokens))

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

    # ----------------------------------------------------
    # EXTRA NAME KEYS
    # ----------------------------------------------------

    if name_alnum:

        keys.add((
            "N_SUFFIX",
            f"N_SUFFIX|{country}|"
            f"{name_alnum[-8:]}"
        ))

        keys.add((
            "N_LENGTH_PREFIX",
            f"N_LENGTH_PREFIX|{country}|"
            f"{len(name_alnum)}:{name_alnum[:4]}"
        ))

    # ----------------------------------------------------
    # ADDRESS
    # ----------------------------------------------------

    for token in address_tokens:

        keys.add((
            "A_TOKEN",
            f"A_TOKEN|{country}|{token}"
        ))

    if len(address_tokens) >= 2:

        pair = sorted([
            address_tokens[0],
            address_tokens[1]
        ])

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
                f"{first_number}|"
                f"{address_tokens[0]}"
            ))

    # ----------------------------------------------------
    # NAME + ADDRESS
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # ADDRESS ALPHANUMERIC
    # ----------------------------------------------------

    if address_alnum:

        keys.add((
            "A_ALNUM_PREFIX",
            f"A_ALNUM_PREFIX|{country}|"
            f"{address_alnum[:12]}"
        ))

        keys.add((
            "A_ALNUM_SUFFIX",
            f"A_ALNUM_SUFFIX|{country}|"
            f"{address_alnum[-12:]}"
        ))

        keys.add((
            "COUNTRY_ADDRESS",
            f"COUNTRY_ADDRESS|{country}|"
            f"{address_alnum[:20]}"
        ))
    if name_translit:

        keys.add((
            "N_TRANSLIT",
            f"N_TRANSLIT|{country}|{name_translit}"
    ))

    if len(name_translit_alnum) >= 5:

        keys.add((
            "N_TRANSLIT_PREFIX",
            f"N_TRANSLIT_PREFIX|{country}|"
            f"{name_translit_alnum[:6]}"
        ))

    for token in translit_tokens:

        keys.add((
            "N_TRANSLIT_TOKEN",
            f"N_TRANSLIT_TOKEN|{country}|{token}"
        ))

    return keys

def load_s1_sample():

    s1 = pl.read_csv(
        S1_PATH,
        separator="\t",
        infer_schema_length=10000,
    )

    return s1.sample(
        n=min(S1_SAMPLE_SIZE, s1.height),
        seed=SEED,
    )


def load_ground_truth(s1_ids):

    gt = pl.read_csv(
        GT_PATH,
        separator="\t",
        infer_schema_length=10000,
    )

    gt = gt.filter(
        pl.col("source1_entity_id").cast(pl.String).is_in(s1_ids)
    )

    positive_pairs = set()

    for row in gt.iter_rows(named=True):

        s1_id = str(row["source1_entity_id"])
        matched = row["matched_entity_ids"]

        if matched is None:
            continue

        if isinstance(matched, str):

            matched = matched.strip()

            if not matched:
                continue

            matched_ids = [
                x.strip()
                for x in matched.split(",")
                if x.strip()
            ]

        elif isinstance(matched, list):

            matched_ids = matched

        else:
            continue

        for s2_id in matched_ids:

            s2_id = str(s2_id)

            if not s2_id.startswith("S2-"):
                continue

            positive_pairs.add(
                (
                    s1_id,
                    s2_id,
                )
            )

    return positive_pairs


def main():

    print("=" * 80)
    print("FAST BLOCK CAP RECALL TEST")
    print("=" * 80)

    # ---------------------------------------------------------
    # S1 sample
    # ---------------------------------------------------------

    print("\nLoading S1 sample...")

    s1 = load_s1_sample()

    print(
        f"S1 sample: {s1.height:,}"
    )

    s1_ids = {
        str(x)
        for x in s1["entity_id"].to_list()
    }

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    print("\nLoading ground truth...")

    positive_pairs = load_ground_truth(
        s1_ids
    )

    print(
        "Ground-truth S1->S2 positive pairs: "
        f"{len(positive_pairs):,}"
    )

    # ---------------------------------------------------------
    # Build S1 key map
    # ---------------------------------------------------------

    key_to_s1 = defaultdict(list)

    for row in s1.iter_rows(named=True):

        s1_id = str(row["entity_id"])

        for key_type, key in make_keys(row):

            key_to_s1[
                (key_type, key)
            ].append(s1_id)

    print(
        "Unique blocking keys: "
        f"{len(key_to_s1):,}"
    )

    # ---------------------------------------------------------
    # First S2 pass:
    # calculate block frequencies
    # ---------------------------------------------------------

    print(
        "\nScanning S2 in chunks "
        "to calculate block frequencies..."
    )

    key_frequency = defaultdict(int)

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

                if full_key in key_to_s1:

                    key_frequency[full_key] += 1

        processed += batch.height

        if processed % 500_000 < BATCH_SIZE:

            print(
                f"Processed: {processed:,}"
            )

    print(
        f"Finished first pass: {processed:,}"
    )

    # ---------------------------------------------------------
    # Second S2 pass:
    # only inspect ground-truth S2 IDs
    # ---------------------------------------------------------

    print(
        "\nChecking positive-pair recall "
        "under caps..."
    )

    positive_by_s2 = defaultdict(set)

    for s1_id, s2_id in positive_pairs:

        positive_by_s2[s2_id].add(
            s1_id
        )

    recovered_pairs = set()

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

            s2_id = str(row["entity_id"])

            if s2_id not in positive_by_s2:
                continue

            for key_type, key in make_keys(row):

                full_key = (
                    key_type,
                    key,
                )

                if full_key not in key_to_s1:
                    continue

                frequency = key_frequency.get(
                    full_key,
                    0,
                )

                if frequency > CAPS[key_type]:
                    continue

                for s1_id in key_to_s1[full_key]:

                    pair = (
                        s1_id,
                        s2_id,
                    )

                    if pair in positive_pairs:

                        recovered_pairs.add(
                            pair
                        )

        processed += batch.height

        if processed % 500_000 < BATCH_SIZE:

            print(
                f"Checking: {processed:,}"
            )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    missed_pairs = (
        positive_pairs
        - recovered_pairs
    )

    recall = (
        len(recovered_pairs)
        / len(positive_pairs)
        if positive_pairs
        else 0
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"Total positive pairs: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Recovered: "
        f"{len(recovered_pairs):,}"
    )

    print(
        f"Missed: "
        f"{len(missed_pairs):,}"
    )

    print(
        f"Blocking recall: "
        f"{recall:.4%}"
    )

    if missed_pairs:

        print(
            "\nFirst 30 missed positive pairs"
        )
        print("-" * 80)

        for s1_id, s2_id in list(
            missed_pairs
        )[:30]:

            print(
                f"S1-{s1_id} -> "
                f"S2-{s2_id}"
            )

    print("\n" + "=" * 80)
    print("CAPS USED")
    print("=" * 80)

    for key_type, cap in CAPS.items():

        print(
            f"{key_type:<18}: "
            f"{cap:,}"
        )


if __name__ == "__main__":
    main()