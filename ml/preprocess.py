"""Text preprocessing, URL extraction and lightweight language detection.

Used by BOTH training and inference so the two never drift apart.
"""
import re

URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"']+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|in|net|org|xyz|top|info|link|co|ly|me|click|online|site|icu|cc|test|example|invalid)\b(?:/[^\s<>\"']*)?"
)

# Small Tamil-English (Tanglish) lexicon: romanised Tamil -> English hint token.
# The hint is APPENDED so the original token is also kept for char n-grams.
CODE_MIX = {
    "pannunga": "do", "panunga": "do", "pannungal": "do", "seiyungal": "do",
    "aagidum": "will_happen", "aagum": "will_happen", "aagividum": "will_happen",
    "udane": "immediately", "udanae": "immediately", "ippove": "immediately",
    "ungal": "your", "unga": "your", "ungaludaya": "your", "ungalukku": "you",
    "kidaikum": "you_get", "kidaikkum": "you_get", "vazhangapadum": "given",
    "anuppunga": "send", "anupunga": "send", "kodunga": "give", "thaanga": "give",
    "nambar": "number", "illai": "not", "illana": "otherwise", "illaiyel": "otherwise",
    "kattanam": "fee", "panam": "money", "velai": "job", "sambalam": "salary",
    "vangi": "bank", "kanakku": "account", "mudakkapadum": "blocked",
    "nirutha": "stop", "seyal": "action", "ippodhu": "now", "inge": "here",
    "click": "click", "link": "link",
}
TANGLISH_MARKERS = set(CODE_MIX) | {"da", "nga", "sir", "ungalukku", "romba", "seekiram", "kandippa"}
TAMIL_SCRIPT = re.compile(r"[\u0B80-\u0BFF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def extract_urls(text: str) -> list:
    """Return URL-like strings found in text (trailing punctuation stripped). Never fetched."""
    found = []
    for m in URL_RE.finditer(text or ""):
        u = m.group(0).rstrip(".,;:!?)]}'\"")
        if u and u not in found:
            found.append(u)
    return found


def detect_language(text: str) -> str:
    """Heuristic detector. Supports: English, Tamil-English (code-mixed), Tamil, Hindi-script, Other.

    Only English and Tamil-English are actually tested in this project.
    """
    if not text:
        return "unknown"
    if TAMIL_SCRIPT.search(text):
        return "tamil"
    if DEVANAGARI.search(text):
        return "hindi_script"
    tokens = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for t in tokens if t in TANGLISH_MARKERS and t not in {"click", "link", "sir", "da"})
    if hits >= 1:
        return "tamil_english"
    return "english"


def normalize(text: str) -> str:
    """Lowercase, mask URLs/numbers, and append English hints for known code-mixed words."""
    t = (text or "").lower()
    t = URL_RE.sub(" urltoken ", t)
    t = re.sub(r"(?:₹|rs\.?|inr)\s?\d[\d,]*", " amounttoken ", t)
    t = re.sub(r"\b\d{4,}\b", " numtoken ", t)
    t = re.sub(r"\b\d+\b", " n ", t)
    words = []
    for w in re.findall(r"[\w\u0B80-\u0BFF']+", t):
        words.append(w)
        if w in CODE_MIX and CODE_MIX[w] != w:
            words.append(CODE_MIX[w])
    return re.sub(r"\s+", " ", " ".join(words)).strip()
