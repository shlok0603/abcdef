from pathlib import Path
import polars as pl


# ============================================================
# Amazon ML Challenge 2026
# Analyze REAL positive entity matches
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "student_resource" / "dataset" / "train"

S1_FILE = TRAIN / "train_source1.tsv"
S2_FILE = TRAIN / "train_source2.tsv"
S3_FILE = TRAIN / "train_source3.tsv"
GT_FILE = TRAIN / "train_ground_truth.tsv"


def load_source(path):
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
    )


def main():

    print("=" * 80)
    print("POSITIVE MATCH ANALYSIS")
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

    # Pick only entities with matches
    gt = gt.filter(
        pl.col("matched_entity_ids").is_not_null()
        & (pl.col("matched_entity_ids").str.strip_chars() != "")
    )

    # Take a manageable sample
    gt_sample = gt.sample(
        n=min(1000, gt.height),
        seed=42,
    )

    print(f"Positive S1 entities available: {gt.height:,}")
    print(f"Analysing sample: {gt_sample.height:,}")

    # --------------------------------------------------------
    # Load sources
    # --------------------------------------------------------

    print("\nLoading source files...")

    s1 = load_source(S1_FILE)
    s2 = load_source(S2_FILE)
    s3 = load_source(S3_FILE)

    print(f"S1 rows: {s1.height:,}")
    print(f"S2 rows: {s2.height:,}")
    print(f"S3 rows: {s3.height:,}")

    # --------------------------------------------------------
    # Convert ground truth into individual pairs
    # --------------------------------------------------------

    pairs = (
        gt_sample
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("matched_list")
        )
        .explode("matched_list")
        .rename({"matched_list": "matched_entity_id"})
    )

    # --------------------------------------------------------
    # Split S2 and S3 matches
    # --------------------------------------------------------

    s2_pairs = pairs.filter(
        pl.col("matched_entity_id").str.starts_with("S2-")
    )

    s3_pairs = pairs.filter(
        pl.col("matched_entity_id").str.starts_with("S3-")
    )

    print("\nPositive pair sample:")
    print(f"S1 → S2 pairs: {s2_pairs.height:,}")
    print(f"S1 → S3 pairs: {s3_pairs.height:,}")

    # --------------------------------------------------------
    # Join S1 → S2
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Join S1 → S3
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Display S1 → S2 examples
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("S1 → S2 POSITIVE MATCHES")
    print("=" * 80)

    print(
        s1_s2.select(
            [
                "source1_entity_id",
                "business_name",
                "business_address",
                "country",
                "matched_entity_id",
                pl.col("business_name_s2").alias("matched_business_name"),
                pl.col("business_address_s2").alias(
                    "matched_business_address"
                ),
                pl.col("country_s2").alias("matched_country"),
            ]
        )
        .head(30)
    )

    # --------------------------------------------------------
    # Display S1 → S3 examples
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("S1 → S3 POSITIVE MATCHES")
    print("=" * 80)

    print(
        s1_s3.select(
            [
                "source1_entity_id",
                "business_name",
                "business_address",
                "country",
                "matched_entity_id",
                pl.col("business_name_s3").alias("matched_business_name"),
                pl.col("business_address_s3").alias(
                    "matched_business_address"
                ),
                pl.col("country_s3").alias("matched_country"),
            ]
        )
        .head(30)
    )

    # --------------------------------------------------------
    # Missing address statistics among positive pairs
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MISSING ADDRESS AMONG POSITIVE PAIRS")
    print("=" * 80)

    s2_missing = s1_s2.filter(
        pl.col("business_address_s2").is_null()
    ).height

    s3_missing = s1_s3.filter(
        pl.col("business_address_s3").is_null()
    ).height

    print(
        f"S1 → S2 positive pairs with missing S2 address: "
        f"{s2_missing:,} / {s1_s2.height:,}"
    )

    print(
        f"S1 → S3 positive pairs with missing S3 address: "
        f"{s3_missing:,} / {s1_s3.height:,}"
    )

    # --------------------------------------------------------
    # Country consistency
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("COUNTRY CONSISTENCY")
    print("=" * 80)

    s2_country_mismatch = s1_s2.filter(
        pl.col("country") != pl.col("country_s2")
    ).height

    s3_country_mismatch = s1_s3.filter(
        pl.col("country") != pl.col("country_s3")
    ).height

    print(
        f"S1 → S2 country mismatches: "
        f"{s2_country_mismatch:,} / {s1_s2.height:,}"
    )

    print(
        f"S1 → S3 country mismatches: "
        f"{s3_country_mismatch:,} / {s1_s3.height:,}"
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()