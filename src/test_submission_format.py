"""Small validator contract test using synthetic test-source files."""

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util

from run_matching_experiment import write_submission_files
import polars as pl


VALIDATOR = Path(__file__).resolve().parents[1] / "student_resource" / "utils" / "validate_submission.py"
spec = importlib.util.spec_from_file_location("validate_submission", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validator)


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        test_dir = root / "dataset" / "test"
        output_dir = root / "output"
        test_dir.mkdir(parents=True)
        for source, rows in {
            "test_source1.tsv": [("S1-1", "A", "", "US"), ("S1-2", "B", "", "France")],
            "test_source2.tsv": [("S2-1", "A", "", "US")],
            "test_source3.tsv": [("S3-1", "B", "", "France")],
        }.items():
            pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row").write_csv(test_dir / source, separator="\t")
        candidates = pl.DataFrame({"source1_entity_id": ["S1-1", "S1-2"], "candidate_entity_id": ["S2-1", "S3-1"]})
        scored = [
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "final_score": 0.9},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S3-1", "final_score": 0.1},
        ]
        matching, candidate = write_submission_files(["S1-1", "S1-2"], candidates, scored, 0.5, output_dir)
        errors, warnings = validator.validate(str(matching), str(candidate), str(test_dir), check_ids=True)
        assert not errors, (errors, warnings)
        assert warnings == []


if __name__ == "__main__":
    main()