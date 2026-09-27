from pathlib import Path
import polars as pl
import re

from normalize import (
    normalize_text,
    accent_folded_text,
    alphanumeric_form,
    token_form,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"

SAMPLE_SIZE = 200


# ------------------------------------------------------------
# Address stopwords
# ------------------------------------------------------------

GENERIC_ADDRESS_WORDS = {
    "road",
    "rd",
    "street",
    "st",
    "drive",
    "dr",
    "lane",
    "ln",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "place",
    "pl",
    "floor",
    "ground",
    "unit",
    "apt",
    "apartment",
    "building",
    "block",
    "no",
    "number",
    "door",
    "shop",
    "india",
    "usa",
    "us",
    "new",
}


def clean_tokens(text):

    if not text:
        return []

    result = []

    for token in token_form(text):

        token = token.strip()

        if len(token) < 3:
            continue

        result.append(token)

    return result


def address_tokens(text):

    if not text:
        return []

    result = []

    for token in token_form(text):

        token = token.strip()

        if len(token) < 4:
            continue

        if token in GENERIC_ADDRESS_WORDS:
            continue

        result.append(token)

    return result


# ------------------------------------------------------------
# Better number extraction
# ------------------------------------------------------------

def normalized_numbers(text):

    if not text:
        return set()

    text = normalize_text(text)

    # Extract individual numeric components.
    raw_numbers = re.findall(
        r"\d+",
        text,
    )

    result = set()

    for value in raw_numbers:

        # Remove leading zeroes.
        value = value.lstrip("0")

        if not value:
            value = "0"

        result.add(value)

    return result


# ------------------------------------------------------------
# New blocking keys
# ------------------------------------------------------------

def new_keys(country, name, address):

    name_norm = normalize_text(name)
    name_folded = accent_folded_text(name)

    name_tokens = clean_tokens(name_norm)

    address_norm = normalize_text(address)
    addr_tokens = address_tokens(address_norm)

    nums = normalized_numbers(address_norm)

    keys = set()

    # ========================================================
    # NAME
    # ========================================================

    # Exact name
    if name_norm:
        keys.add(
            (
                "N_EXACT",
                f"{country}|{name_norm}",
            )
        )

    # Folded name
    if name_folded:
        keys.add(
            (
                "N_FOLDED",
                f"{country}|{name_folded}",
            )
        )

    # Every informative name token
    for token in name_tokens:

        keys.add(
            (
                "N_TOKEN",
                f"{country}|{token}",
            )
        )

    # Adjacent token pairs
    for i in range(len(name_tokens) - 1):

        a = name_tokens[i]
        b = name_tokens[i + 1]

        pair = tuple(sorted((a, b)))

        keys.add(
            (
                "N_PAIR",
                f"{country}|{pair[0]}|{pair[1]}",
            )
        )

    # ========================================================
    # ADDRESS
    # ========================================================

    # Every informative address token
    for token in addr_tokens:

        keys.add(
            (
                "A_TOKEN",
                f"{country}|{token}",
            )
        )

    # Adjacent address token pairs
    for i in range(len(addr_tokens) - 1):

        a = addr_tokens[i]
        b = addr_tokens[i + 1]

        pair = tuple(sorted((a, b)))

        keys.add(
            (
                "A_PAIR",
                f"{country}|{pair[0]}|{pair[1]}",
            )
        )

    # Every normalized number
    for number in nums:

        keys.add(
            (
                "A_NUMBER",
                f"{country}|{number}",
            )
        )

    # Number + informative address token
    for number in nums:

        for token in addr_tokens:

            keys.add(
                (
                    "A_NUM_TOKEN",
                    f"{country}|{number}|{token}",
                )
            )

    # ========================================================
    # HYBRID
    # ========================================================

    for number in nums:

        for token in name_tokens:

            keys.add(
                (
                    "N_TOKEN_NUMBER",
                    f"{country}|{token}|{number}",
                )
            )

    return keys


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 80)
    print("NEW BLOCKING KEY DIAGNOSTIC")
    print("=" * 80)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

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
    # Pairs
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

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    s1_ids = gt[
        "source1_entity_id"
    ].to_list()

    s2_ids = (
        s2_pairs[
            "matched_entity_id"
        ]
        .unique()
        .to_list()
    )

    # --------------------------------------------------------
    # Load only required records
    # --------------------------------------------------------

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
        .collect()
    )

    s1_dict = {
        row["entity_id"]: row
        for row in s1.iter_rows(named=True)
    }

    s2_dict = {
        row["entity_id"]: row
        for row in s2.iter_rows(named=True)
    }

    # --------------------------------------------------------
    # Find pairs missed by old blocker
    # --------------------------------------------------------

    missed = []

    for pair in s2_pairs.iter_rows(named=True):

        s1_row = s1_dict[
            pair["source1_entity_id"]
        ]

        s2_row = s2_dict[
            pair["matched_entity_id"]
        ]

        # New keys
        k1 = new_keys(
            s1_row["country"],
            s1_row["business_name"],
            s1_row["business_address"],
        )

        k2 = new_keys(
            s2_row["country"],
            s2_row["business_name"],
            s2_row["business_address"],
        )

        shared = k1.intersection(k2)

        if not shared:

            missed.append(
                (
                    pair["source1_entity_id"],
                    pair["matched_entity_id"],
                    s1_row,
                    s2_row,
                )
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    total = s2_pairs.height

    recovered = total - len(missed)

    print(
        f"\nTotal positive pairs: {total:,}"
    )

    print(
        f"Recovered by NEW keys: "
        f"{recovered:,}"
    )

    print(
        f"Still missed: "
        f"{len(missed):,}"
    )

    print(
        f"\nNEW KEY RECALL: "
        f"{recovered / total:.4%}"
    )

    # --------------------------------------------------------
    # Remaining failures
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("REMAINING FAILURES")
    print("=" * 80)

    for (
        s1_id,
        s2_id,
        s1_row,
        s2_row,
    ) in missed:

        print("\n" + "-" * 80)

        print("S1:", s1_id)
        print("S2:", s2_id)

        print(
            "S1 NAME:",
            s1_row["business_name"],
        )

        print(
            "S2 NAME:",
            s2_row["business_name"],
        )

        print(
            "S1 ADDRESS:",
            s1_row["business_address"],
        )

        print(
            "S2 ADDRESS:",
            s2_row["business_address"],
        )

        print(
            "S1 NAME TOKENS:",
            clean_tokens(
                normalize_text(
                    s1_row["business_name"]
                )
            ),
        )

        print(
            "S2 NAME TOKENS:",
            clean_tokens(
                normalize_text(
                    s2_row["business_name"]
                )
            ),
        )

        print(
            "S1 ADDRESS TOKENS:",
            address_tokens(
                normalize_text(
                    s1_row["business_address"]
                )
            ),
        )

        print(
            "S2 ADDRESS TOKENS:",
            address_tokens(
                normalize_text(
                    s2_row["business_address"]
                )
            ),
        )

        print(
            "S1 NUMBERS:",
            normalized_numbers(
                s1_row["business_address"]
            ),
        )

        print(
            "S2 NUMBERS:",
            normalized_numbers(
                s2_row["business_address"]
            ),
        )

    print("\n")
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()