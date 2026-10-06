# Pitch scripts and 8-slide PPT outline (no fabricated metrics; quote numbers only from ml/artifacts/metrics.json and say "synthetic data")

## 30-second pitch
Scams are now polished, multilingual and mass-produced, and most detectors just say "scam". ScamShield AI says **why**. It combines ML, security rules and offline URL analysis, highlights the exact evidence, rates risk, and tells users what to verify - and says "uncertain" when it isn't sure. Detect. Explain. Protect.

## 1-minute pitch
Problem: AI-written scams are fluent and personalised, so "bad grammar" no longer works. Gap: black-box detectors give a label users can't verify and miss code-mixed text like Tamil-English. Solution: ScamShield AI - TF-IDF + Logistic Regression for a transparent baseline, a rule engine for urgency/threats/OTP requests/payment demands, an offline URL analyzer, and a weak AI-generated-content signal, fused in a risk engine. Output: risk level, category, highlighted evidence, URL risk, and safe next steps. Honest: trained on synthetic data, we report real metrics and limits.

## 3-minute pitch
PROBLEM (30s): scams impersonating banks, employers, UPI. → GAP (30s): label-only detectors, English-centric, no evidence. → SOLUTION (45s): architecture walk-through. → INNOVATION (30s): explain + evidence + code-mixed + URL + uncertainty. → LIVE DEMO (30s): bank/KYC scam, then the legitimate OTP message to show it does not flag everything. → IMPACT (15s): users learn to spot red flags, fewer clicks on malicious links.

## 5-minute demo flow
1. Home page (privacy note, samples marked simulated). 2. Bank/KYC → show highlights, evidence cards, URL indicators. 3. Job scam (fee request). 4. UPI scam (PIN request). 5. Tamil-English example. 6. **Legitimate OTP message → LOW**. 7. Evaluation page: show real metrics, state they are synthetic and discuss false positives/negatives. 8. Basic-vs-ScamShield and Responsible-AI pages.

## PPT outline
1. **ScamShield AI** · Team Safeguard · AURA-7.1 · Detect. Explain. Protect.
2. **Exact problem** - AI-generated, multilingual scams/phishing; need to separate legitimate, traditional phishing and AI-style scams.
3. **Gap + solution** - label-only black boxes vs hybrid explainable detector.
4. **Innovation** - evidence highlighting, uncertainty-aware output, code-mixed handling, offline URL analysis, weak AI-content signal (not a detector claim).
5. **Architecture + AI/ML** - pipeline diagram; TF-IDF + LogReg; rules; URL; hybrid weights.
6. **Live demo** - bank/KYC, Tamil-English, legitimate message.
7. **Implementation + evaluation + impact** - 28 tests; metrics from metrics.json labelled "synthetic"; limitations; impact on users.
8. **Future scope + conclusion** - public datasets, more languages, OCR, API, extension.
