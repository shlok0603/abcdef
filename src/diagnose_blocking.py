from pathlib import Path
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

SAMPLE_SIZE = 200


# ============================================================
# HELPERS
# ============================================================

def get_tokens(text):

    if not text:
        return []

    return [
        x for x in token_form(text)
        if len(x) >= 3
    ]


def make_keys(country, name, address):

    name_norm = normalize_text(name)
    name_folded = accent_folded_text(name)
    name_alnum = alphanumeric_form(name)

    address_norm = normalize_text(address)

    name_tokens = get_tokens(name_norm)
    address_tokens = get_tokens(address_norm)

    number_sig = number_signature(address_norm)

    keys = set()

    # 1. Exact name
    if name_norm:
        keys.add(
            (
                "NAME_EXACT",
                f"{country}|{name_norm}",
            )
        )

    # 2. Accent folded name
    if name_folded:
        keys.add(
            (
                "NAME_FOLDED",
                f"{country}|{name_folded}",
            )
        )

    # 3. First two meaningful name tokens
    if len(name_tokens) >= 2:

        pair = tuple(sorted(name_tokens[:2]))

        keys.add(
            (
                "NAME_TOKEN2",
                f"{country}|{pair[0]}|{pair[1]}",
            )
        )

    # 4. Name prefix
    if len(name_alnum) >= 6:

        keys.add(
            (
                "NAME_PREFIX6",
                f"{country}|{name_alnum[:6]}",
            )
        )

    # 5. Address number
    if number_sig:

        first_number = number_sig.split()[0]

        keys.add(
            (
                "ADDRESS_NUMBER",
                f"{country}|{first_number}",
            )
        )

    # 6. Name + number
    if name_alnum and number_sig:

        first_number = number_sig.split()[0]

        keys.add(
            (
                "NAME_NUMBER",
                f"{country}|{name_alnum[:6]}|{first_number}",
            )
        )

    # 7. Address token pair
    if len(address_tokens) >= 2:

        pair = tuple(sorted(address_tokens[:2]))

        keys.add(
            (
                "ADDRESS_TOKEN2",
                f"{country}|{pair[0]}|{pair[1]}",
            )
        )

    return keys


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BLOCKING FAILURE DIAGNOSTIC")
    print("=" * 80)

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    print("\nLoading ground truth...")

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

    # --------------------------------------------------------
    # Extract S1 → S2 positive pairs
    # --------------------------------------------------------

    pairs = (
        gt
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

    s2_pairs = pairs.filter(
        pl.col("matched_entity_id")
        .str.starts_with("S2-")
    )

    print(
        f"Sample S1 entities: {gt.height:,}"
    )

    print(
        f"S1 → S2 positive pairs: "
        f"{s2_pairs.height:,}"
    )

    # --------------------------------------------------------
    # Load sampled S1
    # --------------------------------------------------------

    s1_ids = gt["source1_entity_id"].to_list()

    print("\nLoading S1 records...")

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
        .select([
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ])
        .collect()
    )

    # --------------------------------------------------------
    # Load ONLY positive S2 records
    # --------------------------------------------------------

    s2_ids = (
        s2_pairs["matched_entity_id"]
        .unique()
        .to_list()
    )

    print(
        f"\nLoading only {len(s2_ids):,} "
        f"positive S2 records..."
    )

    s2 = (
        pl.scan_csv(
            S2_FILE,
            separator="\t",
            infer_schema_length=1000,
        )
        .filter(
            pl.col("entity_id")
            .is_in(s2_ids)
        )
        .select([
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ])
        .collect()
    )

    print(
        f"S1 loaded: {s1.height:,}"
    )

    print(
        f"S2 positive records loaded: "
        f"{s2.height:,}"
    )

    # --------------------------------------------------------
    # Create dictionaries
    # --------------------------------------------------------

    s1_dict = {
        row["entity_id"]: row
        for row in s1.iter_rows(named=True)
    }

    s2_dict = {
        row["entity_id"]: row
        for row in s2.iter_rows(named=True)
    }

    # --------------------------------------------------------
    # Diagnose every positive pair
    # --------------------------------------------------------

    missed_by_keys = []
    recovered_by_keys = []

    key_type_counts = {
        "NAME_EXACT": 0,
        "NAME_FOLDED": 0,
        "NAME_TOKEN2": 0,
        "NAME_PREFIX6": 0,
        "ADDRESS_NUMBER": 0,
        "NAME_NUMBER": 0,
        "ADDRESS_TOKEN2": 0,
    }

    for pair in s2_pairs.iter_rows(named=True):

        s1_id = pair["source1_entity_id"]
        s2_id = pair["matched_entity_id"]

        s1_row = s1_dict[s1_id]
        s2_row = s2_dict[s2_id]

        s1_keys = make_keys(
            s1_row["country"],
            s1_row["business_name"],
            s1_row["business_address"],
        )

        s2_keys = make_keys(
            s2_row["country"],
            s2_row["business_name"],
            s2_row["business_address"],
        )

        shared = s1_keys.intersection(s2_keys)

        if shared:

            recovered_by_keys.append(
                (s1_id, s2_id, shared)
            )

            for key_type, _ in shared:
                key_type_counts[key_type] += 1

        else:

            missed_by_keys.append(
                (s1_id, s2_id)
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    total = s2_pairs.height

    key_recovered = len(recovered_by_keys)
    key_missed = len(missed_by_keys)

    print("\n")
    print("=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)

    print(
        f"\nTotal positive pairs: "
        f"{total:,}"
    )

    print(
        f"Pairs sharing at least one blocking key: "
        f"{key_recovered:,}"
    )

    print(
        f"Pairs sharing NO blocking key: "
        f"{key_missed:,}"
    )

    print(
        f"\nTHEORETICAL BLOCKING RECALL "
        f"(without 300-record cap): "
        f"{key_recovered / total:.4%}"
    )

    print("\n")
    print("=" * 80)
    print("KEY CONTRIBUTION")
    print("=" * 80)

    for key_type, count in sorted(
        key_type_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        print(
            f"{key_type:<20}"
            f"{count:>8,}"
        )

    # --------------------------------------------------------
    # Show missed pairs
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("POSITIVE PAIRS WITH NO SHARED BLOCKING KEY")
    print("=" * 80)

    for s1_id, s2_id in missed_by_keys[:20]:

        s1_row = s1_dict[s1_id]
        s2_row = s2_dict[s2_id]

        print("\n" + "-" * 80)

        print("S1:", s1_id)
        print("S2:", s2_id)

        print(
            "\nS1 NAME:",
            s1_row["business_name"],
        )

        print(
            "S2 NAME:",
            s2_row["business_name"],
        )

        print(
            "\nS1 ADDRESS:",
            s1_row["business_address"],
        )

        print(
            "S2 ADDRESS:",
            s2_row["business_address"],
        )

        print(
            "\nCOUNTRY:",
            s1_row["country"],
            "/",
            s2_row["country"],
        )

    print("\n")
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()