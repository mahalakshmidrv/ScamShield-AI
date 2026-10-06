"""Print the saved evaluation (re-run `python -m ml.train` to regenerate from scratch)."""
import json
from app.config.settings import ARTIFACTS


def load_metrics() -> dict:
    p = ARTIFACTS / "metrics.json"
    if not p.exists():
        raise FileNotFoundError("metrics.json missing. Run: python -m ml.train")
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    m = load_metrics()
    print(m["data_origin"]); print("Dataset size:", m["dataset_size"], m["split_sizes"])
    print("Class distribution (test):", m["class_distribution"]["test"])
    print("\nPer-class (test):")
    for lab in m["labels"]:
        r = m["test_report"][lab]
        print(f"  {lab:22s} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f} n={int(r['support'])}")
    print("\nBinary scam-vs-legit (test):", m["binary_scam_vs_legitimate_test"])
    print("Per-language (test):", json.dumps(m["per_language_test"], indent=1))
    print("\nConfusion matrix labels:", m["labels"])
    for row in m["confusion_matrix"]: print(" ", row)
