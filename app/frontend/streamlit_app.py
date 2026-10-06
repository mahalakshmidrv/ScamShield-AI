"""ScamShield AI - Streamlit dashboard.  Run from project root:  streamlit run app/frontend/streamlit_app.py

Privacy behaviour (matches the UI statement): submitted text is analysed in memory for the current
session only. Nothing is written to disk or logged by this app. Session counters live in st.session_state.
"""
import html
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.services.analyzer import analyze, InputError
from app.utils.upload import validate_upload, UploadError, OCR_AVAILABLE
from ml.predict import ModelNotTrainedError
from ml.evaluate import load_metrics

st.set_page_config(page_title="ScamShield AI", page_icon="🛡️", layout="wide")

SAMPLES = {  # SIMULATED examples only (reserved example domains)
    "Bank/KYC Example": "Dear customer, your SBI account will be blocked today. Update your KYC immediately: http://sbi-kyc-update.example.com/verify",
    "Job Scam Example": "Congratulations Arun! You are selected for work from home at TechNova Solutions. Pay a refundable registration fee of Rs. 999 to confirm your joining. https://bit.ly/3xJobNow",
    "UPI Scam Example": "You have received a collect request of Rs. 5,000. Approve it and enter your UPI PIN to receive the refund within 10 minutes.",
    "Tamil-English Example": "Your bank account block aagidum, immediately verify pannunga: http://hdfc-kyc-update.example.com/login",
    "Legitimate Example": "Your OTP is 482913. It is valid for 10 minutes. Do not share this OTP with anyone, including bank staff.",
}
COLORS = {"HIGH": "#ff4b4b", "MEDIUM": "#ffa421", "LOW": "#21c46b", "UNCERTAIN": "#8892b0"}
SEV_COLOR = {"high": "#ff4b4b", "medium": "#ffa421", "low": "#7aa2f7", "info": "#8892b0"}

st.markdown("""<style>
.block-container{padding-top:1.5rem}
.ss-card{border:1px solid #2b3245;border-radius:12px;padding:14px 16px;background:#11151f;margin-bottom:10px}
.ss-big{font-size:2rem;font-weight:800;line-height:1.1}
.ss-lbl{font-size:.75rem;letter-spacing:.08em;color:#8892b0;text-transform:uppercase}
.ss-msg{background:#0b0e15;border:1px solid #2b3245;border-radius:10px;padding:14px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.ss-msg mark{background:#ffa42133;color:#ffd8a0;border-bottom:2px solid #ffa421;border-radius:3px;padding:0 2px}
.ss-pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.75rem;font-weight:700;color:#0b0e15}
</style>""", unsafe_allow_html=True)

if "counts" not in st.session_state:
    st.session_state.counts = {"n": 0, "levels": Counter(), "cats": Counter(), "langs": Counter(), "inds": Counter()}
if "msg" not in st.session_state:
    st.session_state.msg = ""


def highlight_html(text, highlights):
    """Escape first, then wrap evidence spans. Input is never rendered as raw HTML."""
    spans = sorted([(s, e) for s, e, _ in highlights if s is not None])
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(text[pos:s])); out.append("<mark>" + html.escape(text[s:e]) + "</mark>"); pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out)


def card(label, value, color="#e6edf3", small=False):
    size = "1.15rem" if small else "2rem"
    st.markdown(f"<div class='ss-card'><div class='ss-lbl'>{html.escape(label)}</div>"
                f"<div class='ss-big' style='color:{color};font-size:{size}'>{html.escape(str(value))}</div></div>", unsafe_allow_html=True)


def record(res):
    c = st.session_state.counts
    c["n"] += 1; c["levels"][res["risk"]["level"]] += 1; c["cats"][res["category"]] += 1; c["langs"][res["language"]] += 1
    for i in {x["indicator"] for x in res["indicators"]}:
        c["inds"][i] += 1


def show_result(text, res):
    lvl = res["risk"]["level"]; col = COLORS[lvl]
    st.markdown("### 🔍 Result")
    a, b, c, d = st.columns(4)
    with a: card("Risk level", lvl, col)
    with b: card("Category", res["category"], small=True)
    with c:
        n_ind = len({i["indicator"] for i in res["indicators"]})
        card("Security indicators", n_ind)
    with d:
        if not res["urls"]: card("URL analysis", "Not provided", small=True)
        else:
            worst = max(res["urls"], key=lambda u: u["risk_score"])
            card("URL analysis", worst["verdict"], "#ffa421" if worst["risk_score"] >= 0.3 else "#21c46b", small=True)
    if res["risk"]["uncertain"]:
        st.warning(res["risk"]["note"])
    elif res["risk"].get("note"):
        st.info(res["risk"]["note"])

    l, r = st.columns([3, 2])
    with l:
        st.markdown("#### Why did ScamShield flag this?")
        if res["highlights"]:
            st.markdown(f"<div class='ss-msg'>{highlight_html(text, res['highlights'])}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ss-msg'>{html.escape(text)}</div>", unsafe_allow_html=True)
        st.write(res["explanation"])
        seen = set()
        for i in res["indicators"]:
            if i["indicator"] in seen: continue
            seen.add(i["indicator"])
            same = [x["evidence"] for x in res["indicators"] if x["indicator"] == i["indicator"]]
            st.markdown(f"<div class='ss-card'><span class='ss-pill' style='background:{SEV_COLOR[i['severity']]}'>{i['severity'].upper()}</span> "
                        f"<b>{html.escape(i['indicator'])}</b><br><i>“{html.escape('”, “'.join(dict.fromkeys(same)))}”</i><br>"
                        f"<span style='color:#9aa5c0'>{html.escape(i['explanation'])}</span></div>", unsafe_allow_html=True)
        if not seen:
            st.success("No security indicators matched in the text.")
    with r:
        st.markdown("#### Model assessment")
        if res["ml"]:
            st.write(f"Top model class: **{res['ml']['top_category']}** (probability {res['ml']['confidence']:.2f})")
            st.write(f"Model probability of any scam class: **{res['ml']['p_scam']:.2f}**")
            st.caption("Probabilities come from a logistic-regression model trained on SYNTHETIC data; treat as a signal, not proof.")
            st.bar_chart({k: v for k, v in res["ml"]["top3"]}, horizontal=True)
        else:
            st.write("URL-only input: the text model was not used.")
        st.markdown("#### Risk score breakdown")
        st.write(f"Combined score: **{res['risk']['score']:.2f}**  (weights: {res['risk']['weights']})")
        st.caption("Weights are hand-set and documented, not tuned on data.")
        st.markdown("#### Language")
        st.write(f"Detected: **{res['language']}**" + (" (code-mixed normalisation applied)" if res["language"] == "tamil_english" else ""))
        if res["ai_content"]:
            ai = res["ai_content"]
            st.markdown("#### AI-generated / scaled content signals")
            st.write(f"**{ai['label']}**")
            for h in ai["indicators"]:
                st.markdown(f"- **{h['indicator']}**: {h['explanation']}")
            st.caption(ai["disclaimer"])

    if res["urls"]:
        st.markdown("#### URL risk indicators (offline analysis, link never opened)")
        for u in res["urls"]:
            with st.expander(f"{u['url']}  -  risk {u['risk_score']:.2f}", expanded=True):
                for i in u["indicators"]:
                    st.markdown(f"- **{i['indicator']}** ({i['severity']}): {i['explanation']}")
                if not u["indicators"]: st.write("No indicators found.")
                st.json(u["features"], expanded=False)
    st.markdown("#### ✅ Recommended action")
    st.success(res["action"])
    st.caption("ScamShield gives a risk assessment, not a verdict. Verify high-impact requests through official channels.")


def page_analyze():
    st.title("🛡️ SCAMSHIELD AI")
    st.markdown("**Analyze suspicious communication before you act.**  ·  *Detect. Explain. Protect.*")
    st.caption("Team Safeguard  ·  CRYPTAURA 2.0  ·  Problem ID AURA-7.1")
    st.markdown("**Simulated examples** (all links use reserved example domains):")
    cols = st.columns(len(SAMPLES))
    for col, (name, txt) in zip(cols, SAMPLES.items()):
        if col.button(name, use_container_width=True):
            st.session_state.msg = txt
    text = st.text_area("Paste a suspicious message (SMS / email / social) or enter a URL", key="msg", height=150,
                        help="Do not paste real OTPs, passwords or card numbers.")
    up = st.file_uploader("Screenshot upload (OCR extension - not enabled in this build)", type=["png", "jpg", "jpeg"])
    if up is not None:
        try:
            validate_upload(up.name, up.size)
        except UploadError as e:
            st.info(str(e))
    if st.button("Analyze Threat", type="primary"):
        try:
            with st.spinner("Analyzing..."):
                res = analyze(text)
            record(res)
            show_result(text.strip(), res)
        except InputError as e:
            st.error(str(e))
        except ModelNotTrainedError as e:
            st.error(str(e))
        except Exception:
            st.error("Something went wrong while analyzing. Please try again with different input.")


def page_eval():
    st.title("📊 Evaluation")
    try:
        m = load_metrics()
    except FileNotFoundError as e:
        st.error(str(e)); return
    st.warning(m["data_origin"] + " Template-level split: test clauses are never seen in training. Real-world performance will differ.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset size", m["dataset_size"]); c2.metric("Train / Val / Test", " / ".join(str(v) for v in m["split_sizes"].values()))
    c3.metric("Test macro-F1", f"{m['test_report']['macro avg']['f1-score']:.3f}")
    b = m["binary_scam_vs_legitimate_test"]
    st.markdown("#### Scam vs legitimate (test set)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Precision (scam)", b["precision_scam"]); d2.metric("Recall (scam)", b["recall_scam"]); d3.metric("False positives", b["fp_legit_flagged_as_scam"]); d4.metric("False negatives", b["fn_scam_missed_as_legit"])
    st.caption("False positive = a legitimate message flagged as scam (erodes trust, may block real bank/job messages). False negative = a scam missed (user is exposed).")
    st.markdown("#### Per-class results (test)")
    rows = [{"class": l, "precision": round(m["test_report"][l]["precision"], 3), "recall": round(m["test_report"][l]["recall"], 3),
             "f1": round(m["test_report"][l]["f1-score"], 3), "support": int(m["test_report"][l]["support"])} for l in m["labels"]]
    st.dataframe(rows, use_container_width=True)
    st.markdown("#### Confusion matrix (rows = true, columns = predicted)")
    try:
        import plotly.express as px
        fig = px.imshow(m["confusion_matrix"], x=m["labels"], y=m["labels"], text_auto=True, color_continuous_scale="Blues", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.dataframe(m["confusion_matrix"])
    st.markdown("#### Class distribution")
    st.bar_chart(m["class_distribution"]["test"], horizontal=True)
    st.markdown("#### Per-language (test)")
    st.json(m["per_language_test"], expanded=False)
    st.markdown("#### Model comparison (validation macro-F1)")
    st.json(m["model_comparison_on_validation"])


def page_analytics():
    st.title("📈 Session analytics")
    c = st.session_state.counts
    if c["n"] == 0:
        st.info("No messages analysed in this session yet. Analyze something first. Nothing is stored between sessions."); return
    st.caption("Counts below are from YOUR current session only (held in memory, not saved).")
    st.metric("Messages analysed this session", c["n"])
    for title, key in (("Risk levels", "levels"), ("Categories", "cats"), ("Languages", "langs"), ("Most common indicators", "inds")):
        if c[key]:
            st.markdown(f"#### {title}"); st.bar_chart(dict(c[key]), horizontal=True)


def page_compare():
    st.title("⚖️ Basic detector vs ScamShield")
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='ss-card'><div class='ss-lbl'>Basic detector</div><div class='ss-big'>“SCAM”</div><p style='color:#9aa5c0'>One label. No reasons. The user must just trust it.</p></div>", unsafe_allow_html=True)
    with b:
        st.markdown("""<div class='ss-card'><div class='ss-lbl'>ScamShield</div><div class='ss-big' style='color:#21c46b'>Potential phishing</div>
        <ul style='color:#c9d3ee'><li>Here are the <b>indicators</b> (urgency, threat, sensitive request…)</li><li>Here is the <b>evidence</b> highlighted in the message</li>
        <li>Here is the <b>URL risk</b> (offline analysis)</li><li>Here is what you should <b>verify</b></li><li>Says <b>“uncertain”</b> when it is not sure</li></ul></div>""", unsafe_allow_html=True)
    st.markdown("**Pipeline:** Input → Validation → Language detection → Preprocessing → NLP features → ML + Rules + URL → Hybrid risk → Evidence → Explanation → Dashboard")


def page_privacy():
    st.title("🔒 Privacy & Responsible AI")
    st.markdown("""**Privacy.** Your submitted message is analysed in memory for this session. This app does **not** write submitted text to disk, a database or logs.
Session analytics are counters held in your browser session and disappear when it ends. Please do not paste real OTPs, passwords or card numbers.
(If a hosting platform adds its own request logging, that is outside this app's control.)

**Responsible AI principles**
- No guarantee of fraud detection; results are *risk assessments* using words like “potential” and “suspicious indicators”.
- AI-generated-content signals are weak heuristics and **cannot prove** text was written by AI.
- No automatic accusation of individuals; the tool never contacts or reports anyone.
- No collection of private credentials. Links are analysed offline and never opened.
- Explainable, uncertainty-aware output (shows “Uncertain” when evidence conflicts or confidence is low).
- Human verification via official channels is recommended for high-impact decisions.
- Tested languages: **English and Tamil-English (code-mixed) only**, on synthetic data.""")


PAGES = {"Analyze": page_analyze, "Evaluation": page_eval, "Session analytics": page_analytics,
         "Basic vs ScamShield": page_compare, "Privacy & Responsible AI": page_privacy}
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("ScamShield AI · Team Safeguard · AURA-7.1")
PAGES[choice]()
