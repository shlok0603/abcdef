from pathlib import Path
import polars as pl


# ============================================================
# Amazon ML Challenge 2026
# Ground Truth Analysis
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


def main():
    print("=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    print(f"\nFile: {GROUND_TRUTH}")

    # Lazy loading
    df = pl.read_csv(
        GROUND_TRUTH,
        separator="\t",
        infer_schema_length=1000,
    )

    print(f"\nRows: {df.height:,}")
    print(f"Columns: {df.columns}")

    # --------------------------------------------------------
    # Identify empty / singleton records
    # --------------------------------------------------------

    df = df.with_columns(
        pl.col("matched_entity_ids")
        .fill_null("")
        .alias("matched_entity_ids")
    )

    df = df.with_columns(
        pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
        .then(0)
        .otherwise(
            pl.col("matched_entity_ids")
            .str.split(",")
            .list.len()
        )
        .alias("match_count")
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MATCH COUNT STATISTICS")
    print("=" * 80)

    stats = df.select(
        [
            pl.col("match_count").min().alias("min"),
            pl.col("match_count").max().alias("max"),
            pl.col("match_count").mean().alias("mean"),
            pl.col("match_count").median().alias("median"),
        ]
    )

    print(stats)

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("MATCH COUNT DISTRIBUTION")
    print("=" * 80)

    distribution = (
        df.group_by("match_count")
        .agg(pl.len().alias("source1_count"))
        .sort("match_count")
    )

    distribution = distribution.with_columns(
        (
            pl.col("source1_count") / df.height * 100
        ).alias("percentage")
    )

    print(distribution)

    # --------------------------------------------------------
    # Singleton analysis
    # --------------------------------------------------------

    singleton_count = df.filter(
        pl.col("match_count") == 0
    ).height

    non_singleton_count = df.filter(
        pl.col("match_count") > 0
    ).height

    print("\n" + "=" * 80)
    print("SINGLETON ANALYSIS")
    print("=" * 80)

    print(f"Singletons:     {singleton_count:,}")
    print(f"Non-singletons: {non_singleton_count:,}")
    print(
        f"Singleton %:    "
        f"{singleton_count / df.height * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Match IDs by source
    # --------------------------------------------------------

    df = df.with_columns(
        pl.col("matched_entity_ids")
        .str.split(",")
        .alias("match_list")
    )

    source_stats = df.select(
        [
            pl.col("match_list")
            .list.eval(
                pl.element().str.starts_with("S2-")
            )
            .list.sum()
            .alias("s2_matches"),

            pl.col("match_list")
            .list.eval(
                pl.element().str.starts_with("S3-")
            )
            .list.sum()
            .alias("s3_matches"),
        ]
    )

    print("\n" + "=" * 80)
    print("S2 / S3 MATCH COUNTS")
    print("=" * 80)

    print(
        "Total S2 matches:",
        source_stats["s2_matches"].sum()
    )

    print(
        "Total S3 matches:",
        source_stats["s3_matches"].sum()
    )

    # --------------------------------------------------------
    # S1 entities with S2 / S3 / both
    # --------------------------------------------------------

    df = df.with_columns(
        [
            pl.col("match_list")
            .list.eval(
                pl.element().str.starts_with("S2-")
            )
            .list.any()
            .alias("has_s2"),

            pl.col("match_list")
            .list.eval(
                pl.element().str.starts_with("S3-")
            )
            .list.any()
            .alias("has_s3"),
        ]
    )

    both = df.filter(
        pl.col("has_s2") & pl.col("has_s3")
    ).height

    only_s2 = df.filter(
        pl.col("has_s2") & ~pl.col("has_s3")
    ).height

    only_s3 = df.filter(
        ~pl.col("has_s2") & pl.col("has_s3")
    ).height

    neither = df.filter(
        ~pl.col("has_s2") & ~pl.col("has_s3")
    ).height

    print("\n" + "=" * 80)
    print("SOURCE COMBINATION DISTRIBUTION")
    print("=" * 80)

    print(f"S2 only:        {only_s2:,}")
    print(f"S3 only:        {only_s3:,}")
    print(f"Both S2 + S3:   {both:,}")
    print(f"Neither:        {neither:,}")

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLES")
    print("=" * 80)

    print("\nSingleton examples:")
    print(
        df.filter(pl.col("match_count") == 0)
        .select(["source1_entity_id", "matched_entity_ids"])
        .head(5)
    )

    print("\nSingle-match examples:")
    print(
        df.filter(pl.col("match_count") == 1)
        .select(["source1_entity_id", "matched_entity_ids"])
        .head(5)
    )

    print("\nMultiple-match examples:")
    print(
        df.filter(pl.col("match_count") >= 3)
        .select(
            [
                "source1_entity_id",
                "matched_entity_ids",
                "match_count",
            ]
        )
        .sort("match_count", descending=True)
        .head(5)
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()