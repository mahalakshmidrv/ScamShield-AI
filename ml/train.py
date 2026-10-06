"""Train the baseline classifier (TF-IDF + Logistic Regression) and save artifacts + REAL metrics.

Run:  python -m ml.train
Pipeline: normalize -> TF-IDF (word 1-2 + char_wb 2-5) -> LogisticRegression (multi-class).
Split is template-level (see build_dataset.py) so test templates are never seen in training.
"""
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from app.config.settings import ARTIFACTS, DATA_PROCESSED
from ml.preprocess import normalize
from ml import build_dataset

SEED = 42


def make_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])


def binary_stats(y_true, y_pred):
    t = [c != "legitimate" for c in y_true]
    p = [c != "legitimate" for c in y_pred]
    tp = sum(a and b for a, b in zip(t, p)); tn = sum((not a) and (not b) for a, b in zip(t, p))
    fp = sum((not a) and b for a, b in zip(t, p)); fn = sum(a and (not b) for a, b in zip(t, p))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "tn": tn, "fp_legit_flagged_as_scam": fp, "fn_scam_missed_as_legit": fn,
            "precision_scam": round(prec, 4), "recall_scam": round(rec, 4), "f1_scam": round(f1, 4)}


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    csv = DATA_PROCESSED / "scam_dataset.csv"
    if not csv.exists():
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        build_dataset.build().to_csv(csv, index=False, encoding="utf-8")
    df = pd.read_csv(csv)
    df["clean"] = df["text"].map(normalize)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))

    vec = make_vectorizer().fit(tr["clean"])          # fit on TRAIN only -> no leakage
    Xtr, Xva, Xte = (vec.transform(d["clean"]) for d in (tr, va, te))

    candidates = {
        "logreg": LogisticRegression(max_iter=3000, C=10, class_weight="balanced", random_state=SEED),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=SEED),
        "complement_nb": ComplementNB(),
    }
    comparison = {}
    for name, m in candidates.items():
        m.fit(Xtr, tr["category"])
        comparison[name] = {"val_macro_f1": round(f1_score(va["category"], m.predict(Xva), average="macro", zero_division=0), 4)}

    clf = candidates["logreg"]   # chosen for calibrated probabilities + transparency (not just best score)
    pred = clf.predict(Xte)
    labels = sorted(df["category"].unique())
    report = classification_report(te["category"], pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(te["category"], pred, labels=labels)

    lang_metrics = {}
    for lang, g in te.groupby("language"):
        gp = clf.predict(vec.transform(g["clean"]))
        lang_metrics[lang] = {"n_test": int(len(g)),
                              "macro_f1": round(f1_score(g["category"], gp, average="macro", zero_division=0), 4),
                              "accuracy": round(float((g["category"].values == gp).mean()), 4),
                              "binary": binary_stats(list(g["category"]), list(gp))}

    metrics = {
        "data_origin": "SYNTHETIC (template-generated). Not representative of real-world accuracy.",
        "dataset_size": int(len(df)),
        "split_sizes": {k: int(len(v)) for k, v in (("train", tr), ("val", va), ("test", te))},
        "class_distribution": {k: {c: int(n) for c, n in v["category"].value_counts().items()}
                               for k, v in (("train", tr), ("val", va), ("test", te))},
        "model_comparison_on_validation": comparison,
        "chosen_model": "logreg (TF-IDF word+char)",
        "test_report": report,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "binary_scam_vs_legitimate_test": binary_stats(list(te["category"]), list(pred)),
        "per_language_test": lang_metrics,
        "templates_in_test_never_seen_in_train": bool(set(te.template_id).isdisjoint(set(tr.template_id))),
    }
    joblib.dump(vec, ARTIFACTS / "vectorizer.joblib")
    joblib.dump(clf, ARTIFACTS / "classifier.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved vectorizer.joblib, classifier.joblib, metrics.json to", ARTIFACTS)
    print("Validation comparison:", comparison)
    print("Test macro-F1:", round(report["macro avg"]["f1-score"], 4), "| weighted-F1:", round(report["weighted avg"]["f1-score"], 4))
    print("Binary (scam vs legit) on test:", metrics["binary_scam_vs_legitimate_test"])


if __name__ == "__main__":
    main()
