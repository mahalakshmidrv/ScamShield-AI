"""Security indicator engine. Returns STRUCTURED evidence (with character spans for highlighting)."""
import re

NEGATION_BEFORE = re.compile(
    r"(do not|don't|dont|never|not to|won't|will never|yaarukum|share pannathinga|pannathinga)[^.\n]{0,40}$", re.I)

# (indicator, severity, regex, explanation)
_RULES = [
    ("Urgency", "medium",
     r"\b(immediately|urgent(?:ly)?|act now|right away|today only|expires? (?:soon|today)|within \d+ ?(?:hours?|hrs?|minutes?|mins?)|last (?:chance|warning)|asap|udane|udanae|ippove|do it today)\b",
     "The message pressures the recipient to act quickly, leaving little time to verify."),
    ("Threat / Pressure", "high",
     r"\b(?:will|shall|may) (?:be )?(?:get )?(?:blocked|suspended|closed|deactivated|terminated|frozen|disabled|disconnected|restricted|cut|cancelled|stopped)\b|\blegal action\b|\bpenalty\b|\bface arrest\b|\bblock aagidum\b|\bclose aagidum\b|\b(?:account|card|number) (?:is|has been) (?:blocked|suspended|locked|limited|on hold)\b|\bcase is registered\b|\bcourt notice\b|\bwill be (?:charged|disconnected)\b|\bonly if you\b",
     "The message threatens a negative outcome (blocking, penalty, legal trouble) to force action."),
    ("Sensitive information request", "high",
     r"\b(?:share|send|provide|enter|submit|tell|reply with|confirm|give|anuppunga|anupunga)\b[^.\n]{0,45}\b(?:otp|pin|cvv|password|passcode|card (?:number|details)|bank details|aadhaar|pan(?: card| number)?|login credentials|net ?banking id|username)\b"
     r"|\b(?:otp|upi pin|password|cvv)\b[^.\n]{0,25}\b(?:anuppunga|anupunga|share pannunga)\b"
     r"|\benter your (?:upi )?pin\b|\bshare the (?:screen )?code\b",
     "The message asks for confidential information (OTP, PIN, password, card or ID details) that legitimate organisations do not request this way."),
    ("Payment request", "high",
     r"\b(?:pay|transfer|send|deposit|kattunga)\b[^.\n]{0,35}\b(?:fee|charges?|fine|tax|deposit|amount|registration|processing|clearance|penalty|rs\.?\s?\d|\d+|rupees)\b"
     r"|\b(?:processing|registration|claim|joining|clearance|delivery|re-delivery) (?:fee|charges?)\b|\brefundable (?:security )?deposit\b|\bkattanam\b|\bsecurity deposit\b"
     r"|\bscan (?:the |this )?qr\b|\bcollect request\b|\bapprove (?:the |it|this)\b",
     "The message asks for money, a fee or a payment approval, which is common in scams."),
    ("Impersonation", "low",
     r"\b(?:sbi|hdfc(?: bank)?|icici(?: bank)?|axis bank|canara bank|bank|rbi|income tax(?: department)?|police|cyber police|cbi|customs|trai|courier|fedex|dhl|india post|amazon|customer (?:care|support)|technical support|helpdesk|it helpdesk|hr team|paytm|phonepe|google ?pay|government)\b",
     "The message refers to a well-known organisation. Scammers often impersonate these."),
    ("Link / call-to-action", "medium",
     r"\b(?:click(?: here| the link| below| on)?|tap (?:here|the link)|open the link|follow the link|sign in to|log ?in (?:at|with|to)|visit|download (?:the )?(?:app|apk|tool)|install (?:the |a )?(?:app|apk|anydesk|remote)[^.\n]{0,20})\b",
     "The message pushes the recipient to follow a link or install something."),
    ("Reward / too-good-to-be-true", "medium",
     r"\b(?:congratulations|you (?:have )?won|lucky (?:draw|winner)|winner|prize|lottery|cashback|gift voucher|gift card|free smartphone|guaranteed (?:returns?|profit|income)|double your money|\d+0% returns?|risk-free|zero risk|sure-shot|100% profit|passive income|triple profit|jeythirukkeenga|kidaikum)\b",
     "The message promises an unusual reward or guaranteed gain, a common lure."),
    ("Job-offer lure", "medium",
     r"\b(?:work from home|no interview|no experience|part-time (?:job|typing)|data entry|earn rs\.?\s?[\d,]+ (?:per|daily)|earn [^.\n]{0,15}daily|daily payment|velai kidaikum)\b",
     "The offer looks too easy (no interview/experience, high pay) which is typical of job scams."),
    ("Secrecy / isolation", "medium",
     r"\b(?:do not|don't) (?:tell|inform) anyone\b|\bkeep (?:this )?(?:confidential|secret)\b",
     "Asking the recipient to keep quiet prevents them from getting a second opinion."),
]
_COMPILED = [(i, s, re.compile(p, re.I), e) for i, s, p, e in _RULES]
SEV_W = {"low": 0.15, "medium": 0.30, "high": 0.50}
ACTION_INDICATORS = {"Urgency", "Threat / Pressure", "Sensitive information request",
                     "Payment request", "Link / call-to-action", "Reward / too-good-to-be-true", "Job-offer lure"}


def detect_indicators(text: str) -> list:
    """Return list of {indicator, severity, evidence, explanation, start, end}."""
    found = []
    for name, sev, rx, expl in _COMPILED:
        for m in rx.finditer(text):
            if name == "Sensitive information request" and NEGATION_BEFORE.search(text[max(0, m.start() - 45):m.start()]):
                continue  # e.g. "Do not share your OTP" is protective, not a request
            found.append({"indicator": name, "severity": sev, "evidence": m.group(0).strip(),
                          "explanation": expl, "start": m.start(), "end": m.end()})
    # Impersonation alone is weak evidence (legit bank SMS mention banks). Keep only with another action signal.
    names = {f["indicator"] for f in found}
    if "Impersonation" in names and not (names & ACTION_INDICATORS):
        found = [f for f in found if f["indicator"] != "Impersonation"]
    return sorted(found, key=lambda f: f["start"])


def rule_score(indicators: list) -> float:
    """0..1 from the strongest severity per unique indicator (transparent, additive, capped)."""
    best = {}
    for f in indicators:
        best[f["indicator"]] = max(best.get(f["indicator"], 0), SEV_W[f["severity"]])
    return min(1.0, sum(best.values()) / 1.2)
