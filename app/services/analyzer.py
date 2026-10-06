"""Orchestrates the full pipeline: validate -> language -> preprocess -> ML + rules + URL -> risk -> explanation."""
import re
from app.config.settings import MAX_CHARS, load_categories
from app.rules.indicators import detect_indicators, rule_score
from app.services.url_analyzer import analyze_url
from app.services.ai_content import analyze_ai_content
from app.services.risk_engine import combine
from ml.preprocess import extract_urls, detect_language
from ml import predict as ml_predict

CATS = load_categories()
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class InputError(ValueError):
    """Raised with a user-friendly message for invalid input."""


def validate(text) -> str:
    if text is None or not isinstance(text, str):
        raise InputError("Please provide text to analyze.")
    text = _CONTROL.sub("", text).strip()
    if not text:
        raise InputError("The input is empty. Paste a message or a URL to analyze.")
    if len(text) > MAX_CHARS:
        raise InputError(f"The input is too long ({len(text)} characters). The limit is {MAX_CHARS}.")
    return text


def _looks_like_url_only(text: str) -> bool:
    urls = extract_urls(text)
    return len(urls) == 1 and " " not in text and len(urls[0]) >= len(text) - 2


def _explain(level, cat_label, inds, url_res, ml_note) -> str:
    names, seen = [], set()
    for i in inds:
        if i["indicator"] not in seen:
            seen.add(i["indicator"]); names.append(i["indicator"].lower())
    parts = []
    if names:
        parts.append("The message shows: " + ", ".join(names[:5]) + ".")
    bad = [u for u in url_res if u["risk_score"] >= 0.3]
    if bad:
        parts.append("The link has suspicious characteristics: " + ", ".join(i["indicator"].lower() for i in bad[0]["indicators"] if i["severity"] != "info")[:200] + ".")
    if not parts:
        parts.append("No strong scam indicators were found in the text.")
    if ml_note:
        parts.append(ml_note)
    return " ".join(parts)


def analyze(raw_text) -> dict:
    """Main entry point. Raises InputError for invalid input. Never stores or logs the text."""
    text = validate(raw_text)
    lang = detect_language(text)
    urls = extract_urls(text)
    url_results = [analyze_url(u) for u in urls[:5]]
    url_s = max((u["risk_score"] for u in url_results), default=None)

    if _looks_like_url_only(text):
        risk = combine(None, 0.0, url_s)
        inds = []
        for u in url_results:
            inds += [{**i, "start": None, "end": None} for i in u["indicators"] if i["severity"] != "info"]
        return {"input_type": "url", "language": lang, "category_id": "unknown", "category": "URL only (no message text)",
                "ml": None, "indicators": [], "urls": url_results, "ai_content": None, "risk": risk,
                "explanation": _explain(risk["level"], "", [], url_results, ""),
                "action": CATS["phishing"]["action"] if risk["level"] != "LOW" else "No red flags found, but this is not proof of safety. Prefer typing the official address yourself.",
                "highlights": []}

    ml = ml_predict.predict(text)
    inds = detect_indicators(text)
    rs = rule_score(inds)
    risk = combine(ml, rs, url_s)
    ai = analyze_ai_content(text)

    cat_id = ml["category"]
    if risk["level"] == "UNCERTAIN":
        cat_id = "unknown" if ml["confidence"] < 0.40 else cat_id
    elif risk["level"] == "LOW":
        cat_id = "legitimate" if ml["p_legitimate"] >= 0.5 else cat_id
    cat = CATS[cat_id]
    prefix = "Potential " if cat["is_scam"] and risk["level"] in ("HIGH", "MEDIUM", "UNCERTAIN") else ""
    category = prefix + cat["label"]
    if ml["category"] != "legitimate" and risk["level"] == "LOW":
        action = CATS["unknown"]["action"]
    else:
        action = cat["action"]
    if risk["level"] == "UNCERTAIN":
        action = CATS["unknown"]["action"] + " " + (cat["action"] if cat["is_scam"] else "")
    return {"input_type": "text", "language": lang, "category_id": cat_id, "category": category,
            "ml": {"top_category": ml["category"], "confidence": round(ml["confidence"], 3),
                   "p_scam": round(ml["p_scam"], 3),
                   "top3": sorted(ml["probabilities"].items(), key=lambda kv: -kv[1])[:3]},
            "indicators": inds, "urls": url_results, "ai_content": ai, "risk": risk,
            "explanation": _explain(risk["level"], category, inds, url_results, risk["note"]),
            "action": action.strip(),
            "highlights": [(i["start"], i["end"], i["indicator"]) for i in inds]}
