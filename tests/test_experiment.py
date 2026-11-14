"""Software-only tests. Fixtures are NOT research data or a Kaggle benchmark.

These checks intentionally never run run_experiment on artificial data and do
not publish predictive performance on the fixtures as a research result.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import (DATASET_HANDLE, FEATURE_NAMES, LEXICONS, MODEL_NAMES,
                        calculate_metrics, clean_text, cue_matrix,
                        fit_representations, locate_csv, make_splits,
                        paired_group_bootstrap, prepare_frame, template_hash,
                        train_classifier)


def alphabetic_id(index):
    chars = []
    for _ in range(4):
        chars.append(chr(97 + index % 26))
        index //= 26
    return "".join(chars)


def split_fixture():
    # Distinct alphabetic tokens avoid the intentional number-masking grouping.
    raw = pd.DataFrame({"text_combined": [f"fixture message token{alphabetic_id(i)}" for i in range(120)],
                        "label": [i % 2 for i in range(120)]})
    frame, _ = prepare_frame(raw)
    return frame


def test_exact_dataset_identity():
    assert DATASET_HANDLE == "naserabdullahalam/phishing-email-dataset"
    assert len(MODEL_NAMES) == 3


def test_cleaner_removes_scripts_and_html():
    assert clean_text("<script>harmful()</script><p>Hello &amp; goodbye</p>") == "hello & goodbye"


def test_cleaner_normalizes_whitespace_case():
    assert clean_text("  HELLO\n\tWorld ") == "hello world"


def test_cleaner_masks_urls_without_fetching():
    output = clean_text("See https://example.invalid/login?id=3 and help@example.invalid")
    assert "urltoken" in output and "emailtoken" in output
    assert "example.invalid" not in output


def test_cleaner_character_limit():
    assert len(clean_text("a" * 100, max_chars=20)) == 20


def test_feature_shape_and_empty_text():
    features = cue_matrix(["", "hello"])
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.count_nonzero(features) == 0


def test_phrase_boundaries():
    assert cue_matrix(["freeform banknote winnerish todayland"])[0].sum() == 0


def test_urgency_and_fear_interaction():
    features = cue_matrix(["urgent: your account has been suspended"])[0]
    assert features[0] == 1 and features[3] == 1
    assert features[-2] == 1 and features[-1] == 0


def test_phrase_space_hyphen_equivalence():
    assert cue_matrix(["last chance"])[0, 0] == cue_matrix(["last-chance"])[0, 0] == 1


def test_template_groups_number_punctuation_variants():
    assert template_hash(clean_text("Invoice 123, due today!")) == template_hash(clean_text("Invoice 456 due today."))


def test_template_groups_url_variants():
    assert template_hash(clean_text("Visit https://example.invalid/a")) == template_hash(clean_text("Visit https://other.invalid/b"))


def test_duplicate_and_conflict_audit():
    raw = pd.DataFrame({"text_combined": ["hello", " HELLO ", "same", "same", None, ""],
                        "label": [0, 0, 0, 1, 1, 0]})
    frame, audit = prepare_frame(raw)
    assert len(frame) == 1
    assert audit["missing_text_or_label_rows"] == 1
    assert audit["empty_after_cleaning_rows"] == 1
    assert audit["conflicting_rows_removed"] == 2
    assert audit["duplicate_rows_removed"] == 1


def test_unknown_labels_fail_closed():
    with pytest.raises(ValueError):
        prepare_frame(pd.DataFrame({"text_combined": ["x"], "label": ["phishing"]}))


def test_schema_mismatch_fails():
    with pytest.raises(ValueError):
        prepare_frame(pd.DataFrame({"Email Text": ["x"], "Email Type": [1]}))


def test_splits_are_disjoint_and_exhaustive():
    frame = split_fixture()
    parts = make_splits(frame)
    combined = np.concatenate(list(parts.values()))
    assert sorted(combined.tolist()) == list(range(len(frame)))
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(frame.group_hash.iloc[parts[a]]).isdisjoint(frame.group_hash.iloc[parts[b]])


def test_split_is_reproducible():
    frame = split_fixture()
    p1, p2 = make_splits(frame), make_splits(frame)
    assert all(np.array_equal(p1[key], p2[key]) for key in p1)


def test_related_templates_stay_together():
    frame = split_fixture()
    extras = frame.iloc[:15].copy()
    extras["text_hash"] = extras.text_hash + "variant"
    enlarged = pd.concat([frame, extras], ignore_index=True)
    parts = make_splits(enlarged)
    group_partition = {}
    for name, positions in parts.items():
        for group in enlarged.group_hash.iloc[positions]:
            assert group not in group_partition or group_partition[group] == name
            group_partition[group] = name


def test_small_sample_fails():
    with pytest.raises(ValueError):
        make_splits(split_fixture().iloc[:10])


def test_no_test_vocabulary_leakage():
    vectorizer, scaler, train, evaluation = fit_representations(
        ["meeting tomorrow", "meeting today", "urgent account", "urgent request"],
        ["unseenholdoutword urgent"], 50)
    assert "unseenholdoutword" not in vectorizer.vocabulary_
    assert train[2].shape[1] == train[0].shape[1] + len(FEATURE_NAMES)
    assert evaluation[2].shape[1] == train[2].shape[1]


def test_scaler_fits_training_only():
    training = ["meeting tomorrow", "meeting today", "urgent account", "urgent request"]
    _, scaler, _, _ = fit_representations(training, ["urgent urgent urgent"], 50)
    assert np.allclose(scaler.mean_, cue_matrix(training).mean(axis=0))


def test_metric_arithmetic_not_a_benchmark():
    metrics = calculate_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)
    assert metrics["f1_positive"] == 0.5
    assert metrics["accuracy"] == 0.5


def test_paired_bootstrap_identical_predictions():
    y = np.array([0, 1, 0, 1])
    prediction = np.array([0, 1, 1, 1])
    output = paired_group_bootstrap(y, prediction, prediction, np.array(["a", "b", "c", "d"]), 100, 42)
    assert output["observed_delta"] == 0
    assert output["ci_95_percentile"] == [0.0, 0.0]


def test_classifier_api_on_numeric_fixture_only():
    from scipy.sparse import csr_matrix
    model = train_classifier(csr_matrix([[0.], [1.], [2.], [3.]]), np.array([0, 0, 1, 1]), 1.0, 42)
    probabilities = model.predict_proba(csr_matrix([[1.5]]))
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_wrong_filename_not_silently_accepted(tmp_path):
    file = tmp_path / "different_dataset.csv"
    file.write_text("text_combined,label\nexample,0\n")
    with pytest.raises(ValueError):
        locate_csv(str(file))


def test_missing_file_has_no_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        locate_csv(str(tmp_path / "phishing_email.csv"))
