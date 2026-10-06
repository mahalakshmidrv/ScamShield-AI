"""Load saved artifacts and predict. Model is loaded once and cached."""
import joblib
from functools import lru_cache
from app.config.settings import ARTIFACTS
from ml.preprocess import normalize


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load():
    v, c = ARTIFACTS / "vectorizer.joblib", ARTIFACTS / "classifier.joblib"
    if not (v.exists() and c.exists()):
        raise ModelNotTrainedError("Model files not found. Run:  python -m ml.build_dataset && python -m ml.train")
    return joblib.load(v), joblib.load(c)


def predict_proba(text: str) -> dict:
    """Return {category: probability} for the input text."""
    vec, clf = _load()
    probs = clf.predict_proba(vec.transform([normalize(text)]))[0]
    return {c: float(p) for c, p in zip(clf.classes_, probs)}


def predict(text: str) -> dict:
    probs = predict_proba(text)
    top = max(probs, key=probs.get)
    return {"category": top, "confidence": probs[top], "p_legitimate": probs.get("legitimate", 0.0),
            "p_scam": 1.0 - probs.get("legitimate", 0.0), "probabilities": probs}
