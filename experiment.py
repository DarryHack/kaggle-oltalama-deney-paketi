"""Reproducible experiment for naserabdullahalam/phishing-email-dataset.

This program never substitutes another corpus or manufactures research results.
The lexicons below are hand-written proxies, not ground-truth emotion labels.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import platform
import re
import sys
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

DATASET_HANDLE = "naserabdullahalam/phishing-email-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_HANDLE}"
EXPECTED_FILENAME = "phishing_email.csv"
SEED = 42
MODEL_NAMES = ("A_text_only", "B_cues_only", "C_text_and_cues")
C_GRID = (0.1, 1.0, 10.0)

# Frozen before seeing the evaluation data. Keep these lists unchanged during a run.
# Matching words does not establish the sender's intent or the recipient's emotion.
LEXICONS = {
    "urgency": (
        "urgent", "urgently", "immediately", "as soon as possible", "asap",
        "act now", "respond now", "right now", "today", "deadline", "expires",
        "expiring", "last chance", "limited time", "within 24 hours",
        "within 48 hours", "final notice", "time sensitive", "without delay",
    ),
    "fear_threat": (
        "suspended", "suspension", "blocked", "terminated", "termination",
        "unauthorized", "fraud", "compromised", "security breach", "penalty",
        "legal action", "account closure", "account locked", "lose access",
        "identity theft", "warning", "risk", "threat", "danger",
    ),
    "reward": (
        "winner", "won", "prize", "reward", "congratulations", "bonus",
        "lottery", "jackpot", "free", "exclusive offer", "cash reward",
        "claim your", "inheritance", "million dollars", "beneficiary",
        "discount", "gift", "guaranteed", "special offer",
    ),
    "trust_authority": (
        "official", "verified", "secure", "trusted", "security team",
        "support team", "customer support", "administrator", "bank",
        "government", "verify your account", "confirm your identity",
        "account verification", "security department", "authorized",
    ),
}


def _compile_phrases(phrases: tuple[str, ...]) -> re.Pattern:
    alternatives = []
    for phrase in sorted(phrases, key=len, reverse=True):
        alternatives.append(r"[\s_-]+".join(re.escape(p) for p in phrase.split()))
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)", re.I)


PATTERNS = {name: _compile_phrases(words) for name, words in LEXICONS.items()}
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
FEATURE_NAMES = [f"{name}_{stat}" for name in LEXICONS
                 for stat in ("present", "per_100_words", "log_count")]
FEATURE_NAMES += ["urgency_and_fear", "urgency_and_reward"]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: str, max_chars: int = 20_000) -> str:
    """Deterministic text-only preprocessing; never visits URLs or executes HTML."""
    text = unicodedata.normalize("NFKC", html.unescape(str(value)))
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text[:max_chars].strip()


def template_hash(text: str) -> str:
    """Conservative heuristic, not a semantic near-duplicate detector.

    URL/email placeholders were already introduced by clean_text. Number and
    punctuation variants share a group. Paraphrased campaigns can still leak.
    """
    value = re.sub(r"\d+", " numbertoken ", text)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cue_matrix(texts: list[str]) -> np.ndarray:
    features = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float64)
    for row, text in enumerate(texts):
        count_words = max(1, len(re.findall(r"\b\w+\b", text)))
        counts = [len(pattern.findall(text)) for pattern in PATTERNS.values()]
        for index, count in enumerate(counts):
            features[row, index * 3:index * 3 + 3] = (
                float(count > 0), 100.0 * count / count_words, np.log1p(count))
        features[row, -2:] = [float(counts[0] > 0 and counts[1] > 0),
                             float(counts[0] > 0 and counts[2] > 0)]
    return features


def prepare_frame(raw: pd.DataFrame, max_chars: int = 20_000) -> tuple[pd.DataFrame, dict]:
    if not {"text_combined", "label"}.issubset(raw.columns):
        raise ValueError("Beklenen sütunlar: text_combined, label. Başka veri seti kullanılmayacak.")
    frame = raw[["text_combined", "label"]].copy()
    frame["source_row"] = np.arange(len(frame))
    audit = {"raw_rows": len(frame), "input_columns": list(raw.columns),
             "max_chars_after_cleaning": max_chars}
    nulls = frame["text_combined"].isna() | frame["label"].isna()
    audit["missing_text_or_label_rows"] = int(nulls.sum())
    frame = frame.loc[~nulls].copy()
    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    invalid = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
    if invalid.any():
        raise ValueError(f"0/1 dışında veya çözülemeyen {int(invalid.sum())} etiket var; otomatik dönüştürülmedi.")
    frame["label"] = numeric_labels.astype(int)
    audit["nonmissing_input_class_counts"] = {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()}
    audit["raw_rows_longer_than_char_limit"] = int((frame.text_combined.astype(str).str.len() > max_chars).sum())
    frame["text"] = frame.text_combined.astype(str).map(lambda s: clean_text(s, max_chars))
    empty = ~frame.text.str.contains(r"\w", regex=True)
    audit["empty_after_cleaning_rows"] = int(empty.sum())
    frame = frame.loc[~empty].copy()
    frame["text_hash"] = frame.text.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    label_counts = frame.groupby("text_hash").label.nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)
    conflicting = frame.text_hash.isin(conflicting_hashes)
    audit["conflicting_text_groups"] = len(conflicting_hashes)
    audit["conflicting_rows_removed"] = int(conflicting.sum())
    frame = frame.loc[~conflicting].copy()
    audit["duplicate_rows_removed"] = int(frame.duplicated("text_hash").sum())
    frame = frame.drop_duplicates("text_hash").reset_index(drop=True)
    frame["group_hash"] = frame.text.map(template_hash)
    audit["retained_rows"] = len(frame)
    audit["retained_class_counts"] = {str(k): int(v) for k, v in frame.label.value_counts().sort_index().items()}
    audit["template_groups"] = int(frame.group_hash.nunique())
    audit["groups_with_both_labels_retained_together"] = int((frame.groupby("group_hash").label.nunique() > 1).sum())
    audit["language_detection_performed"] = False
    audit["emotion_ground_truth_available"] = False
    return frame[["source_row", "text", "label", "text_hash", "group_hash"]], audit


def make_splits(frame: pd.DataFrame, seed: int = SEED) -> dict[str, np.ndarray]:
    if set(frame.label.unique()) != {0, 1}:
        raise ValueError("Her iki sınıf da gerekli.")
    groups_per_class = frame.groupby("label").group_hash.nunique()
    if groups_per_class.min() < 10:
        raise ValueError("Her sınıfta en az 10 ayrı metin/şablon grubu gerekli.")
    y, groups = frame.label.to_numpy(), frame.group_hash.to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(inner.split(np.zeros(len(dev)), y[dev], groups[dev]))
    partitions = {"train": dev[train_local], "validation": dev[val_local], "test": test}
    names = list(partitions)
    for name, indices in partitions.items():
        if set(y[indices]) != {0, 1}:
            raise ValueError(f"{name} bölmesinde iki sınıf da bulunmuyor; deney durduruldu.")
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            assert not (set(groups[partitions[left]]) & set(groups[partitions[right]])), "Group leakage"
            assert not (set(frame.text_hash.iloc[partitions[left]]) & set(frame.text_hash.iloc[partitions[right]])), "Text leakage"
    return partitions


def locate_csv(explicit: str | None = None, allow_download: bool = False) -> tuple[Path, str]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name != EXPECTED_FILENAME:
            raise ValueError(f"Dosya adı {EXPECTED_FILENAME} olmalı; farklı dosya otomatik kabul edilmiyor.")
        return path, "user_supplied_local_file_provenance_not_independently_verified"
    roots = (Path("/kaggle/input"), Path("data"), Path.cwd())
    candidates = set()
    for root in roots:
        if root.is_dir():
            candidates.update(p.resolve() for p in root.rglob(EXPECTED_FILENAME) if p.is_file())
    if len(candidates) == 1:
        path = next(iter(candidates))
        return path, "mounted_file_schema_checked_origin_requires_dataset_attachment_check"
    if len(candidates) > 1:
        raise ValueError("Birden çok phishing_email.csv bulundu. --csv ile tam dosya yolunu belirtin.")
    if not allow_download:
        raise FileNotFoundError("phishing_email.csv yok. Tam olarak belirtilen Kaggle veri setini ekleyin veya --download kullanın.")
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("İndirme için: python -m pip install kagglehub") from exc
    try:
        directory = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as exc:
        raise RuntimeError(f"{DATASET_HANDLE} indirilemedi. Başka veri veya sahte veri kullanılmadı. Asıl hata: {exc}") from exc
    matches = list(directory.rglob(EXPECTED_FILENAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"Kaggle paketinde tek bir {EXPECTED_FILENAME} bekleniyordu; {len(matches)} bulundu.")
    return matches[0].resolve(), "downloaded_by_kagglehub_exact_dataset_handle"


def fit_representations(train_texts: list[str], eval_texts: list[str], max_features: int):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=1.0,
                               max_features=max_features, sublinear_tf=True,
                               lowercase=False, dtype=np.float64)
    text_train = vectorizer.fit_transform(train_texts)
    text_eval = vectorizer.transform(eval_texts)
    cue_scaler = StandardScaler()
    cue_train = sparse.csr_matrix(cue_scaler.fit_transform(cue_matrix(train_texts)))
    cue_eval = sparse.csr_matrix(cue_scaler.transform(cue_matrix(eval_texts)))
    train_matrices = (text_train, cue_train, sparse.hstack([text_train, cue_train], format="csr"))
    eval_matrices = (text_eval, cue_eval, sparse.hstack([text_eval, cue_eval], format="csr"))
    return vectorizer, cue_scaler, train_matrices, eval_matrices


def train_classifier(X: sparse.spmatrix, y: np.ndarray, C: float, seed: int):
    model = LogisticRegression(C=C, solver="liblinear", class_weight="balanced",
                               max_iter=3000, tol=1e-4, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Model yakınsamadı. Sonuç yazılmadı; max_iter/ölçeklemeyi gözden geçirin.")
    return model


def calculate_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision_positive": float(precision_score(y, prediction, zero_division=0)),
        "recall_positive": float(recall_score(y, prediction, zero_division=0)),
        "f1_positive": float(f1_score(y, prediction, zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision_ap": float(average_precision_score(y, probability)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "test_rows": int(len(y)), "threshold": 0.5,
    }


def paired_group_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_c: np.ndarray,
                           groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Resample template groups, paired across A and C; report positive F1 delta."""
    _, inv = np.unique(groups, return_inverse=True)
    number_groups = int(inv.max()) + 1
    counts = np.zeros((number_groups, 6), dtype=np.int64)
    for offset, pred in ((0, pred_a), (3, pred_c)):
        for j, mask in enumerate(((y == 1) & (pred == 1),
                                  (y == 0) & (pred == 1),
                                  (y == 1) & (pred == 0))):
            counts[:, offset + j] = np.bincount(inv, weights=mask, minlength=number_groups).astype(np.int64)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        totals = counts[rng.integers(0, number_groups, number_groups)].sum(axis=0)
        tpa, fpa, fna, tpc, fpc, fnc = totals
        da, dc = 2 * tpa + fpa + fna, 2 * tpc + fpc + fnc
        if da and dc:
            deltas.append(2 * tpc / dc - 2 * tpa / da)
    observed = float(f1_score(y, pred_c, zero_division=0) - f1_score(y, pred_a, zero_division=0))
    interval = np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else [None, None]
    return {"quantity": "positive_F1_C_minus_A", "observed_delta": observed,
            "ci_95_percentile": interval, "bootstrap_replicates": n_boot,
            "valid_replicates": len(deltas), "resampling_unit": "heuristic_template_group",
            "seed": seed,
            "caution": "Conditional on this split and fixed models; not an external-validity or causality test."}


def make_figures(results_dir: Path, metrics: pd.DataFrame, y: np.ndarray,
                 predictions: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics.model, metrics.f1_positive)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Positive-class F1")
    ax.set_title("Held-out test: model comparison")
    for index, value in enumerate(metrics.f1_positive):
        ax.text(index, value + 0.015, f"{value:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(results_dir / "f1_comparison.png", dpi=180)
    plt.close(fig)
    for model in MODEL_NAMES:
        matrix = confusion_matrix(y, predictions[f"{model}_prediction"], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(matrix)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, str(value), ha="center", va="center",
                    bbox=dict(boxstyle="round", alpha=0.85))
        ax.set_xticks([0, 1], ["Label 0", "Label 1"])
        ax.set_yticks([0, 1], ["Label 0", "Label 1"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Dataset label")
        ax.set_title(model)
        fig.tight_layout()
        fig.savefig(results_dir / f"confusion_{model}.png", dpi=180)
        plt.close(fig)


def run_experiment(csv_path: Path, output: Path, source_mode: str,
                   seed: int = SEED, max_features: int = 50_000,
                   max_chars: int = 20_000, bootstrap: int = 1000,
                   save_models: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Çıktı dizini boş değil: {output}. Yeni bir dizin kullanın; eski sonuçlar ezilmeyecek.")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {
        "status": "started_not_completed", "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_handle": DATASET_HANDLE, "dataset_url": DATASET_URL,
        "input_file_name": csv_path.name, "input_file_bytes": csv_path.stat().st_size,
        "input_sha256": sha256_file(csv_path), "source_mode": source_mode,
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")},
        "configuration": {"seed": seed, "C_grid": list(C_GRID), "max_features": max_features,
                          "max_chars": max_chars, "test_threshold": 0.5,
                          "class_weight": "balanced", "ngram_range": [1, 2],
                          "selection_metric": "validation_macro_F1",
                          "split": "first fold of group-stratified 5-fold outer, 4-fold inner",
                          "bootstrap_replicates": bootstrap},
        "limitations": [
            "Positive class may pool spam, scams and phishing; do not claim pure credential-phishing accuracy.",
            "Emotion cues are investigator-defined English lexicons; no emotion ground truth or causality claim.",
            "Template grouping cannot remove all semantic near duplicates or source-domain leakage.",
            "No independent modern, Turkish, temporal or source-held-out evaluation is performed.",
            "A single fixed grouped split is used; results are split-specific.",
            "Input text is truncated after deterministic normalization at the configured character limit.",
        ],
    }
    write_json(output / "run_manifest.json", manifest)
    try:
        raw = pd.read_csv(csv_path, dtype={"text_combined": "string", "label": "string"}, on_bad_lines="error")
        frame, audit = prepare_frame(raw, max_chars)
        del raw
        write_json(output / "data_audit.json", audit)
        if len(frame) < 100:
            raise ValueError("100'den az kullanılabilir kayıt: araştırma deneyi çalıştırılmadı.")
        splits = make_splits(frame, seed)
        split_table = frame[["source_row", "text_hash", "group_hash", "label"]].copy()
        split_table["split"] = ""
        for name, indices in splits.items():
            split_table.loc[indices, "split"] = name
        split_table.to_csv(output / "split_manifest.csv", index=False)
        manifest["split_counts"] = {name: {"rows": len(indices),
            "label_0": int((frame.label.iloc[indices] == 0).sum()),
            "label_1": int((frame.label.iloc[indices] == 1).sum())} for name, indices in splits.items()}
        manifest["exact_hash_overlap_between_splits"] = 0
        manifest["heuristic_group_overlap_between_splits"] = 0
        print("Data audit:", audit, flush=True)
        print("Split counts:", manifest["split_counts"], flush=True)
        train, validation, test = splits["train"], splits["validation"], splits["test"]
        y = frame.label.to_numpy()
        _, _, X_train, X_val = fit_representations(frame.text.iloc[train].tolist(),
                                                   frame.text.iloc[validation].tolist(), max_features)
        tuning_rows = []
        best_C = {}
        with threadpool_limits(limits=1):
            for index, name in enumerate(MODEL_NAMES):
                candidates = []
                for C in C_GRID:
                    model = train_classifier(X_train[index], y[train], C, seed)
                    prediction = (model.predict_proba(X_val[index])[:, 1] >= 0.5).astype(int)
                    score = float(f1_score(y[validation], prediction, average="macro", zero_division=0))
                    candidates.append((score, C))
                    tuning_rows.append({"model": name, "C": C, "validation_macro_F1": score})
                # Stable tie-break: smaller regularization inverse C wins.
                best_C[name] = sorted(candidates, key=lambda item: (-item[0], item[1]))[0][1]
                print(name, "selected C:", best_C[name], flush=True)
        pd.DataFrame(tuning_rows).to_csv(output / "validation_scores.csv", index=False)
        del X_train, X_val
        development = np.sort(np.concatenate([train, validation]))
        vectorizer, scaler, X_dev, X_test = fit_representations(
            frame.text.iloc[development].tolist(), frame.text.iloc[test].tolist(), max_features)
        predictions = frame.loc[test, ["source_row", "text_hash", "group_hash", "label"]].reset_index(drop=True)
        rows, report = [], {}
        models = {}
        with threadpool_limits(limits=1):
            for index, name in enumerate(MODEL_NAMES):
                model = train_classifier(X_dev[index], y[development], best_C[name], seed)
                probability = model.predict_proba(X_test[index])[:, 1]
                prediction = (probability >= 0.5).astype(int)
                models[name] = model
                predictions[f"{name}_probability"] = probability
                predictions[f"{name}_prediction"] = prediction
                measured = calculate_metrics(y[test], probability)
                rows.append({"model": name, "selected_C": best_C[name], **measured})
                report[name] = classification_report(y[test], prediction, labels=[0, 1],
                                                     target_names=["dataset_label_0", "dataset_label_1"],
                                                     output_dict=True, zero_division=0)
                print(name, measured, flush=True)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(output / "metrics.csv", index=False)
        write_json(output / "metrics.json", rows)
        write_json(output / "classification_reports.json", report)
        predictions.to_csv(output / "test_predictions.csv", index=False)
        difference = paired_group_bootstrap(y[test], predictions.A_text_only_prediction.to_numpy(),
            predictions.C_text_and_cues_prediction.to_numpy(), predictions.group_hash.to_numpy(), bootstrap, seed + 2)
        write_json(output / "paired_f1_comparison.json", difference)
        cues_test = cue_matrix(frame.text.iloc[test].tolist())
        descriptors = []
        for index, name in enumerate(LEXICONS):
            for label in (0, 1):
                subset = cues_test[y[test] == label]
                descriptors.append({"cue": name, "dataset_label": label, "test_rows": len(subset),
                                    "presence_rate": float(subset[:, index * 3].mean()),
                                    "mean_count_per_100_words": float(subset[:, index * 3 + 1].mean())})
        pd.DataFrame(descriptors).to_csv(output / "cue_descriptives_test.csv", index=False)
        pd.DataFrame({"feature": FEATURE_NAMES, "coefficient": models["B_cues_only"].coef_[0]}).to_csv(
            output / "cue_model_coefficients.csv", index=False)
        if save_models:
            import joblib
            joblib.dump({"vectorizer": vectorizer, "cue_scaler": scaler, "models": models,
                         "feature_names": FEATURE_NAMES, "max_chars": max_chars,
                         "warning": "Do not load untrusted joblib/pickle files."}, output / "models.joblib")
        make_figures(output, metrics, y[test], predictions)
        manifest.update({"status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                         "elapsed_seconds": time.perf_counter() - started, "selected_C": best_C,
                         "tfidf_feature_count": len(vectorizer.get_feature_names_out()),
                         "cue_feature_count": len(FEATURE_NAMES)})
        write_json(output / "run_manifest.json", manifest)
        return {"metrics": rows, "audit": audit, "comparison": difference, "manifest": manifest}
    except Exception as exc:
        manifest.update({"status": "failed_not_a_completed_experiment", "error": str(exc),
                         "elapsed_seconds": time.perf_counter() - started})
        write_json(output / "run_manifest.json", manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Exact phishing_email.csv path")
    parser.add_argument("--download", action="store_true", help="Download ONLY the specified Kaggle dataset")
    parser.add_argument("--output", default="results/real_run")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.max_features < 10 or args.max_chars < 100 or args.bootstrap < 100:
        parser.error("max-features >= 10, max-chars >= 100, bootstrap >= 100 olmalı.")
    try:
        path, mode = locate_csv(args.csv, args.download)
        run_experiment(path, Path(args.output), mode, args.seed, args.max_features,
                       args.max_chars, args.bootstrap, args.save_models)
    except Exception as exc:
        print(f"DENEY TAMAMLANMADI: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
