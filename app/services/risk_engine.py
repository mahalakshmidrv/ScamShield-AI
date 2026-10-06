"""Transparent hybrid risk engine: ML + rules + URL. Weights are hand-set, documented, not tuned."""
from app.config.settings import W_WITH_URL, W_NO_URL, HIGH_T, MEDIUM_T, LOW_CONF


def combine(ml: dict | None, rule_s: float, url_s: float | None) -> dict:
    """ml: output of ml.predict.predict (or None for URL-only input)."""
    if ml is None:                               # URL-only: no text model signal
        s = url_s or 0.0
        level = "HIGH" if s >= 0.6 else "MEDIUM" if s >= 0.3 else "LOW"
        return {"score": round(s, 3), "level": level, "uncertain": False, "weights": {"url": 1.0},
                "note": "URL-only analysis. A low score means no red flags were found, not that the link is safe."}
    p = ml["p_scam"]
    w = W_WITH_URL if url_s is not None else W_NO_URL
    score = w["ml"] * p + w["rules"] * rule_s + (w.get("url", 0) * url_s if url_s is not None else 0)
    strongest_external = max(rule_s, url_s or 0.0)
    reasons = []
    if ml["confidence"] < LOW_CONF:
        reasons.append("model confidence is low")
    if p >= 0.6 and strongest_external < 0.15:
        reasons.append("the model suspects a scam but no concrete security indicators were found")
    if p < 0.4 and strongest_external >= 0.6:
        reasons.append("security indicators are strong but the model sees a legitimate pattern")
    level = "HIGH" if score >= HIGH_T else "MEDIUM" if score >= MEDIUM_T else "LOW"
    uncertain = bool(reasons)
    if uncertain:
        level = "UNCERTAIN"
    return {"score": round(score, 3), "level": level, "uncertain": uncertain, "weights": w,
            "note": ("Uncertain — additional verification recommended (" + "; ".join(reasons) + ").") if uncertain else ""}
