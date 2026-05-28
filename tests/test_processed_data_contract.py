from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/liar_binary.csv")


def test_processed_dataset_exists():
    assert DATA_PATH.exists()


def test_processed_dataset_schema_and_labels():
    df = pd.read_csv(DATA_PATH)
    expected_columns = [
        "split",
        "statement_id",
        "original_label",
        "label",
        "label_name",
        "statement",
        "subject",
        "speaker",
        "speaker_job",
        "state_info",
        "party_affiliation",
        "context",
        "text_length_chars",
        "text_length_words",
    ]

    assert list(df.columns) == expected_columns
    assert set(df["split"].unique()) == {"train", "valid", "test"}
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["label_name"].unique()) == {"credible", "not_credible"}
    assert "half-true" not in set(df["original_label"].unique())
    assert df["statement"].notna().all()
    assert (df["text_length_words"] > 0).all()


def test_each_split_contains_both_binary_classes():
    df = pd.read_csv(DATA_PATH)
    split_label_counts = df.groupby("split")["label"].nunique()

    assert (split_label_counts == 2).all()
