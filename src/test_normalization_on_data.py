from pathlib import Path
import polars as pl

from normalize import (
    normalize_text,
    alphanumeric_form,
    sorted_token_form,
    content_token_form,
    number_signature,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"

GT_FILE = TRAIN / "train_ground_truth.tsv"
S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"


def get_source_records(path, ids):
    """
    Memory-safe lookup.

    Instead of loading the entire source into memory, scan it lazily
    and collect only the IDs needed for our sample.
    """
    return (
        pl.scan_csv(
            path,
            separator="\t",
            infer_schema_length=1000,
        )
        .filter(pl.col("entity_id").is_in(ids))
        .collect()
    )


def main():

    print("=" * 80)
    print("REAL DATA NORMALIZATION TEST")
    print("=" * 80)

    print("\nLoading ground truth sample...")

    gt = pl.read_csv(
        GT_FILE,
        separator="\t",
        infer_schema_length=1000,
    )

    gt = gt.filter(
        pl.col("matched_entity_ids").is_not_null()
        & (pl.col("matched_entity_ids").str.strip_chars() != "")
    )

    # Smaller sample so the experiment stays fast.
    gt = gt.sample(
        n=min(200, gt.height),
        seed=42,
    )

    pairs = (
        gt
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("matched_list")
        )
        .explode("matched_list")
        .rename({"matched_list": "matched_entity_id"})
    )

    s2_pairs = pairs.filter(
        pl.col("matched_entity_id").str.starts_with("S2-")
    )

    s3_pairs = pairs.filter(
        pl.col("matched_entity_id").str.starts_with("S3-")
    )

    print(f"Sample S1 entities: {gt.height:,}")
    print(f"S1→S2 positive pairs: {s2_pairs.height:,}")
    print(f"S1→S3 positive pairs: {s3_pairs.height:,}")

    # ------------------------------------------------------------
    # Get only required source records
    # ------------------------------------------------------------

    s1_ids = gt["source1_entity_id"].to_list()
    s2_ids = s2_pairs["matched_entity_id"].to_list()
    s3_ids = s3_pairs["matched_entity_id"].to_list()

    print("\nReading only required records...")

    s1 = get_source_records(S1_FILE, s1_ids)
    s2 = get_source_records(S2_FILE, s2_ids)
    s3 = get_source_records(S3_FILE, s3_ids)

    print(f"S1 records loaded: {s1.height:,}")
    print(f"S2 records loaded: {s2.height:,}")
    print(f"S3 records loaded: {s3.height:,}")

    # ------------------------------------------------------------
    # Join S1 → S2
    # ------------------------------------------------------------

    s1_s2 = (
        s2_pairs
        .join(
            s1,
            left_on="source1_entity_id",
            right_on="entity_id",
            how="left",
        )
        .join(
            s2,
            left_on="matched_entity_id",
            right_on="entity_id",
            how="left",
            suffix="_s2",
        )
    )

    # ------------------------------------------------------------
    # Join S1 → S3
    # ------------------------------------------------------------

    s1_s3 = (
        s3_pairs
        .join(
            s1,
            left_on="source1_entity_id",
            right_on="entity_id",
            how="left",
        )
        .join(
            s3,
            left_on="matched_entity_id",
            right_on="entity_id",
            how="left",
            suffix="_s3",
        )
    )

    # ------------------------------------------------------------
    # Show examples
    # ------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("NORMALIZED S1 → S2 EXAMPLES")
    print("=" * 80)

    rows = s1_s2.head(15).iter_rows(named=True)

    for row in rows:

        s1_name = row["business_name"]
        s2_name = row["business_name_s2"]

        s1_addr = row["business_address"]
        s2_addr = row["business_address_s2"]

        print("\nS1 ID:", row["source1_entity_id"])
        print("S2 ID:", row["matched_entity_id"])

        print("\nNAME")
        print(" S1 raw :", s1_name)
        print(" S2 raw :", s2_name)
        print(" S1 norm:", normalize_text(s1_name))
        print(" S2 norm:", normalize_text(s2_name))
        print(" S1 alnum:", alphanumeric_form(s1_name))
        print(" S2 alnum:", alphanumeric_form(s2_name))
        print(" S1 content:", content_token_form(s1_name))
        print(" S2 content:", content_token_form(s2_name))

        print("\nADDRESS")
        print(" S1 raw :", s1_addr)
        print(" S2 raw :", s2_addr)
        print(" S1 norm:", normalize_text(s1_addr))
        print(" S2 norm:", normalize_text(s2_addr))
        print(" S1 numbers:", number_signature(s1_addr))
        print(" S2 numbers:", number_signature(s2_addr))

    print("\n")
    print("=" * 80)
    print("NORMALIZED S1 → S3 EXAMPLES")
    print("=" * 80)

    rows = s1_s3.head(15).iter_rows(named=True)

    for row in rows:

        s1_name = row["business_name"]
        s3_name = row["business_name_s3"]

        s1_addr = row["business_address"]
        s3_addr = row["business_address_s3"]

        print("\nS1 ID:", row["source1_entity_id"])
        print("S3 ID:", row["matched_entity_id"])

        print("\nNAME")
        print(" S1 raw :", s1_name)
        print(" S3 raw :", s3_name)
        print(" S1 norm:", normalize_text(s1_name))
        print(" S3 norm:", normalize_text(s3_name))
        print(" S1 alnum:", alphanumeric_form(s1_name))
        print(" S3 alnum:", alphanumeric_form(s3_name))
        print(" S1 content:", content_token_form(s1_name))
        print(" S3 content:", content_token_form(s3_name))

        print("\nADDRESS")
        print(" S1 raw :", s1_addr)
        print(" S3 raw :", s3_addr)
        print(" S1 norm:", normalize_text(s1_addr))
        print(" S3 norm:", normalize_text(s3_addr))
        print(" S1 numbers:", number_signature(s1_addr))
        print(" S3 numbers:", number_signature(s3_addr))

    print("\n")
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()