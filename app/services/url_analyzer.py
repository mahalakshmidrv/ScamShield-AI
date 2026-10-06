"""Offline URL analysis. URLs are NEVER fetched or resolved; only the string is inspected."""
import re
from urllib.parse import urlparse, unquote

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "rb.gy", "ow.ly", "shorturl.at", "tiny.cc"}
RISKY_TLDS = {"xyz", "top", "icu", "click", "online", "site", "info", "cc", "link", "live", "work", "support"}
SECOND_LEVEL = {"co", "com", "org", "gov", "ac", "net", "edu"}
# Tiny demo list. NOT exhaustive. Brand token -> official registered domains.
OFFICIAL = {
    "sbi": {"sbi.co.in", "onlinesbi.sbi", "sbi.bank.in"}, "hdfc": {"hdfcbank.com", "hdfc.com"},
    "icici": {"icicibank.com"}, "axis": {"axisbank.com"}, "paytm": {"paytm.com"}, "phonepe": {"phonepe.com"},
    "amazon": {"amazon.in", "amazon.com"}, "google": {"google.com"}, "irctc": {"irctc.co.in"},
    "netflix": {"netflix.com"}, "microsoft": {"microsoft.com"}, "incometax": {"incometax.gov.in"},
    "paypal": {"paypal.com"}, "flipkart": {"flipkart.com"}, "whatsapp": {"whatsapp.com"},
}
RESERVED_SUFFIX = (".example", ".test", ".invalid", ".localhost")
RESERVED_DOMAINS = {"example.com", "example.org", "example.net"}
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "@": "a", "$": "s"})
IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
SUSP_PATH = re.compile(r"(?i)(login|signin|verify|update|secure|account|kyc|confirm|wallet|claim|reward|bank|password)")


def registered_domain(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _ind(name, severity, evidence, explanation):
    return {"indicator": name, "severity": severity, "evidence": evidence, "explanation": explanation}


def analyze_url(url: str) -> dict:
    raw = url.strip()
    has_scheme = bool(re.match(r"(?i)^https?://", raw))
    parsed = urlparse(raw if has_scheme else "//" + raw)
    host = (parsed.hostname or "").lower()
    path = unquote(parsed.path or "")
    inds, feats = [], {}
    feats.update(url_length=len(raw), scheme=(parsed.scheme or "not stated"), host=host,
                 path_depth=len([p for p in path.split("/") if p]), has_ip=bool(IP_RE.match(host)),
                 has_at_symbol="@" in raw, percent_encoded="%" in raw, hyphen_count=host.count("-"),
                 subdomain_count=max(0, len(host.split(".")) - len(registered_domain(host).split("."))) if host else 0)
    reg = registered_domain(host)
    feats["registered_domain"] = reg
    is_reserved = reg in RESERVED_DOMAINS or host.endswith(RESERVED_SUFFIX)

    if feats["has_ip"]:
        inds.append(_ind("IP-based URL", "high", host, "The link uses a raw IP address instead of a domain name, typical of throw-away phishing servers."))
    if host in SHORTENERS:
        inds.append(_ind("URL shortener", "medium", host, "Shortened links hide the real destination."))
    if feats["has_at_symbol"]:
        inds.append(_ind("'@' in URL", "high", "@", "Text before '@' can disguise the real host in a URL."))
    if host.startswith("xn--") or ".xn--" in host:
        inds.append(_ind("Punycode / look-alike characters", "high", host, "Internationalised characters can imitate well-known domains."))
    if feats["subdomain_count"] >= 3:
        inds.append(_ind("Excessive subdomains", "medium", host, "Many subdomains are used to make a fake domain look legitimate."))
    if feats["hyphen_count"] >= 2:
        inds.append(_ind("Many hyphens in domain", "low", host, "Hyphen-stuffed domains (e.g. bank-kyc-update) are common in phishing."))
    if feats["percent_encoded"]:
        inds.append(_ind("URL encoding", "low", "%", "Encoded characters can hide the real path or parameters."))
    if feats["path_depth"] >= 4:
        inds.append(_ind("Deep path", "low", path, "Very deep paths are sometimes used to bury a malicious page."))
    if len(raw) > 75:
        inds.append(_ind("Long URL", "low", f"{len(raw)} characters", "Unusually long URLs can hide suspicious parameters."))
    if parsed.scheme == "http":
        inds.append(_ind("No HTTPS", "medium", "http://", "The link is not encrypted. Legitimate login pages use HTTPS."))
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if tld in RISKY_TLDS:
        inds.append(_ind("Frequently abused TLD", "low", "." + tld, "This domain ending is often used in throw-away scam sites."))
    if SUSP_PATH.search(host + path) and not feats["has_ip"]:
        m = SUSP_PATH.search(host + path)
        inds.append(_ind("Sensitive keyword in URL", "low", m.group(0), "Words like login/verify/kyc are used to look official."))

    norm = host.translate(LEET).replace("-", "").replace(".", "")
    for brand, officials in OFFICIAL.items():
        if brand in norm and reg not in officials:
            inds.append(_ind("Look-alike / brand misuse", "high", f"'{brand}' in {host}",
                             f"The domain contains '{brand}' but is not an official {brand} domain (offline check against a small demo list)."))
            break
    if ("gov" in host.split(".")[0:-1] or "govt" in norm) and not host.endswith((".gov.in", ".nic.in", ".gov")):
        inds.append(_ind("Fake government look", "high", host, "The domain suggests a government site but does not end in .gov.in / .nic.in."))
    if is_reserved:
        inds.append(_ind("Reserved/test domain (simulated)", "info", host, "Reserved domain used for safe demonstrations; it can never be a real site."))

    w = {"low": 0.12, "medium": 0.25, "high": 0.45, "info": 0.0}
    score = min(1.0, sum(w[i["severity"]] for i in inds) / 0.9)
    return {"url": raw, "features": feats, "indicators": inds, "risk_score": round(score, 3),
            "verdict": "Suspicious indicators" if score >= 0.3 else "No strong red flags (not proof of safety)",
            "fetched": False}
