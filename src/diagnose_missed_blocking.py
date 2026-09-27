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

S1_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_source1.tsv"
S2_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_source2.tsv"
GT_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

MISSED = [
    ("S1-846160933", "S2-369008263"),
    ("S1-844661225", "S2-295707247"),
    ("S1-259229061", "S2-678678339"),
    ("S1-361429955", "S2-593489416"),
    ("S1-541880350", "S2-261241593"),
    ("S1-943060701", "S2-938871768"),
    ("S1-703492122", "S2-849746512"),
    ("S1-613345210", "S2-201152052"),
    ("S1-912578795", "S2-364862001"),
    ("S1-259229061", "S2-368679186"),
    ("S1-445318882", "S2-598856967"),
    ("S1-455133974", "S2-832650240"),
]


def keys(row):

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

    result = []

    if name:
        result.append(
            f"N_EXACT|{country}|{name}"
        )

    if folded:
        result.append(
            f"N_FOLDED|{country}|{folded}"
        )

    if name_tokens:
        result.append(
            f"N_TOKEN1|{country}|{name_tokens[0]}"
        )

    if len(name_tokens) >= 2:
        result.append(
            f"N_TOKEN2|{country}|"
            f"{name_tokens[0]}|{name_tokens[1]}"
        )

    if len(name_alnum) >= 5:
        result.append(
            f"N_PREFIX|{country}|{name_alnum[:5]}"
        )

    if numbers:
        result.append(
            f"A_NUM|{country}|"
            f"{numbers.split()[0]}"
        )

    if address_tokens:
        result.append(
            f"A_TOKEN1|{country}|"
            f"{address_tokens[0]}"
        )

    if len(address_tokens) >= 2:
        result.append(
            f"A_TOKEN2|{country}|"
            f"{address_tokens[0]}|"
            f"{address_tokens[1]}"
        )

    if name_alnum and numbers:
        result.append(
            f"NA|{country}|"
            f"{name_alnum[:5]}|"
            f"{numbers.split()[0]}"
        )

    return set(result)


def load_ids(path, ids):

    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
    )

    return df.filter(
        pl.col("entity_id")
        .cast(pl.String)
        .is_in(ids)
    )


def main():

    all_s1 = {
        s1 for s1, s2 in MISSED
    }

    all_s2 = {
        s2 for s1, s2 in MISSED
    }

    s1 = load_ids(
        S1_PATH,
        all_s1
    )

    s2 = load_ids(
        S2_PATH,
        all_s2
    )

    s1_map = {
        str(r["entity_id"]): r
        for r in s1.iter_rows(named=True)
    }

    s2_map = {
        str(r["entity_id"]): r
        for r in s2.iter_rows(named=True)
    }

    print("=" * 80)
    print("MISSED BLOCKING PAIRS")
    print("=" * 80)

    for s1_id, s2_id in MISSED:

        r1 = s1_map[s1_id]
        r2 = s2_map[s2_id]

        k1 = keys(r1)
        k2 = keys(r2)

        shared = sorted(
            k1 & k2
        )

        print("\n" + "-" * 80)

        print(
            f"S1: {s1_id}"
        )

        print(
            f"S2: {s2_id}"
        )

        print(
            f"\nS1 name: "
            f"{r1['business_name']}"
        )

        print(
            f"S2 name: "
            f"{r2['business_name']}"
        )

        print(
            f"\nS1 address: "
            f"{r1['business_address']}"
        )

        print(
            f"S2 address: "
            f"{r2['business_address']}"
        )

        print(
            f"\nCountry: "
            f"{r1['country']} / "
            f"{r2['country']}"
        )

        print(
            "\nShared blocking keys:"
        )

        if shared:
            for k in shared:
                print(
                    f"  {k}"
                )
        else:
            print(
                "  NONE"
            )


if __name__ == "__main__":
    main()