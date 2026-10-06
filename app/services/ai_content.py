"""'Possible AI-generated / scaled content indicators'. A weak heuristic signal, NOT a detector.
Linguistic analysis cannot reliably prove text was written by an AI."""
import re
import statistics

_SIGNALS = [
    ("Generic salutation", r"\bdear (?:valued )?(?:customer|user|client|member|candidate|sir/?madam|account holder)\b",
     "Generic greeting suggests a template sent to many people rather than personal contact."),
    ("Highly formal boilerplate", r"\b(?:we regret to inform you|please be advised|kindly|pursuant to|hereby|we are pleased to inform|we would like to bring to your attention)\b",
     "Polished formal phrasing is common in mass-produced or generated scam text."),
    ("Templated call-to-action", r"\bkindly (?:click|verify|confirm|log ?in|update|follow|visit)\b|\bclick (?:here|the link below) to (?:verify|confirm|update|claim|avoid)\b",
     "A standard 'kindly verify/click to avoid' structure that repeats across scam templates."),
    ("Unfilled placeholder", r"\[(?:name|customer|amount|link)\]|\{(?:name|customer|amount|link)\}|<(?:name|link)>",
     "Template placeholders suggest bulk generation."),
]


def analyze_ai_content(text: str) -> dict:
    hits = []
    for name, rx, expl in _SIGNALS:
        m = re.search(rx, text, re.I)
        if m:
            hits.append({"indicator": name, "evidence": m.group(0), "explanation": expl})
    sents = [s for s in re.split(r"[.!?]\s+", text) if len(s.split()) >= 3]
    if len(sents) >= 4:
        lens = [len(s.split()) for s in sents]
        if statistics.pstdev(lens) < 2.5:
            hits.append({"indicator": "Unnaturally uniform sentences", "evidence": f"{len(sents)} sentences of similar length",
                         "explanation": "Very uniform sentence length can occur in generated or templated text."})
    level = "none" if not hits else "weak" if len(hits) == 1 else "possible"
    return {"level": level, "indicators": hits,
            "label": {"none": "No notable indicators", "weak": "Weak indicator", "possible": "Possible AI-generated/scaled content indicators"}[level],
            "disclaimer": "This is one weak risk signal. It cannot prove that text was AI-generated; human scammers use templates too, and legitimate organisations also write formally."}
