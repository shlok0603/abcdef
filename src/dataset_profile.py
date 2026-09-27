from pathlib import Path
import polars as pl


# ============================================================
# Amazon ML Challenge 2026
# Memory-safe dataset profiling
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "student_resource" / "dataset"

FILES = {
    "train_source1": DATASET / "train" / "train_source1.tsv",
    "train_source2": DATASET / "train" / "train_source2.tsv",
    "train_source3": DATASET / "train" / "train_source3.tsv",
    "train_ground_truth": DATASET / "train" / "train_ground_truth.tsv",
    "test_source1": DATASET / "test" / "test_source1.tsv",
    "test_source2": DATASET / "test" / "test_source2.tsv",
    "test_source3": DATASET / "test" / "test_source3.tsv",
}


def profile_file(name: str, path: Path):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    print(f"Path: {path}")
    print(f"Size: {path.stat().st_size / (1024 ** 2):.2f} MB")

    # Lazy scan: does not load the entire file into memory.
    lf = pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        ignore_errors=False,
    )

    # Schema
    print("\nSchema:")
    print(lf.collect_schema())

    # Row count
    row_count = lf.select(pl.len()).collect().item()
    print(f"\nRows: {row_count:,}")

    # Small sample
    sample = lf.head(5).collect()

    print("\nFirst 5 rows:")
    print(sample)

    # Null counts
    nulls = (
        lf.select(
            [
                pl.col(c).null_count().alias(c)
                for c in lf.collect_schema().names()
            ]
        )
        .collect()
    )

    print("\nNull counts:")
    print(nulls)

    # Unique countries for source files
    columns = lf.collect_schema().names()

    if "country" in columns:
        countries = (
            lf.select(pl.col("country").value_counts())
            .unnest("country")
            .sort("count", descending=True)
            .collect()
        )

        print("\nCountry distribution:")
        print(countries)


def main():
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - DATASET PROFILE")
    print("=" * 80)

    print(f"\nDataset root: {DATASET}")

    for name, path in FILES.items():
        profile_file(name, path)

    print("\n" + "=" * 80)
    print("PROFILE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()