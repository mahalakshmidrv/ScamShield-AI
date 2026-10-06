"""Build the SYNTHETIC training dataset.

IMPORTANT / HONESTY NOTE
- Every row is synthetic: generated from hand-written templates with random slot fills.
- No real user messages, OTPs, passwords or personal data are used.
- Rows from the same template are never split across train/val/test (prevents template leakage).
- Metrics on this data measure performance on THESE templates only. They do NOT predict
  real-world accuracy. Replace/extend with public datasets (see README) for real evaluation.

Run:  python -m ml.build_dataset
"""
import random
import pandas as pd
from app.config.settings import DATA_PROCESSED, DATA_PROCESSED as OUT

SEED = 42
VARIANTS_PER_TEMPLATE = 8

SLOTS = {
    "bank": ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Canara Bank"],
    "amt": ["Rs. 499", "Rs. 1,500", "Rs. 2,999", "Rs. 5,000", "Rs. 25,000"],
    "n": ["2", "4", "6", "12", "24"],
    "name": ["Arun", "Priya", "Karthik", "Divya", "Suresh"],
    "co": ["TechNova Solutions", "BrightPath Infotech", "Zenith Global", "CloudNest Pvt Ltd"],
    "url": ["http://sbi-kyc-update.example.com/verify", "https://bit.ly/3xKycUp", "http://192.0.2.15/login",
            "https://secure-bank-verify.example.net/auth", "http://paytm-rewards.example.xyz/claim", "tinyurl.com/kyc-now"],
    "ourl": ["https://www.example.com/help", "https://www.example.org/track"],
}

# (category, language, source_type, template)
T = [
    # ---- bank_kyc
    ("bank_kyc", "english", "sms", "Dear customer, your {bank} account will be blocked today. Update your KYC immediately: {url}"),
    ("bank_kyc", "english", "sms", "{bank} ALERT: KYC expired. Your account will be suspended within {n} hours. Click {url} to verify PAN and Aadhaar."),
    ("bank_kyc", "english", "email", "We regret to inform you that your {bank} account has been restricted due to incomplete KYC. Kindly confirm your details at {url} to avoid penalty."),
    ("bank_kyc", "english", "sms", "Your {bank} net banking will be deactivated. Complete KYC now or face legal action. {url}"),
    ("bank_kyc", "tamil_english", "sms", "Ungal {bank} account block aagidum, KYC update pannunga udane: {url}"),
    # ---- upi_payment
    ("upi_payment", "english", "sms", "You have received a collect request of {amt}. Approve it and enter your UPI PIN to receive the refund."),
    ("upi_payment", "english", "sms", "Payment of {amt} failed. Scan this QR code and enter your UPI PIN to get your money back. {url}"),
    ("upi_payment", "english", "social", "Hi, I sent you {amt} by mistake. Please approve the request on your UPI app and share the PIN so it gets reversed."),
    ("upi_payment", "english", "sms", "Cashback of {amt} is waiting. Accept the collect request in your payment app within {n} minutes."),
    ("upi_payment", "tamil_english", "sms", "Unga account-ku {amt} cashback kidaikum. UPI PIN anuppunga udane, illana cancel aagidum."),
    # ---- job_internship
    ("job_internship", "english", "email", "Congratulations {name}! You are selected for work from home at {co}. Pay a refundable registration fee of {amt} to confirm your joining."),
    ("job_internship", "english", "social", "Part-time job: earn {amt} daily by liking videos. No interview, no experience needed. Pay {amt} deposit to start. {url}"),
    ("job_internship", "english", "email", "Dear candidate, your internship offer at {co} is ready. Transfer a processing fee of {amt} today to receive the offer letter."),
    ("job_internship", "english", "sms", "Urgent hiring! Data entry job, salary Rs. 40,000 per month. Send {amt} security deposit and your bank details. {url}"),
    ("job_internship", "tamil_english", "social", "Work from home velai, sambalam {amt} daily. Registration kattanam {amt} anuppunga, udane join pannunga."),
    # ---- investment
    ("investment", "english", "social", "Guaranteed {n}0% returns in {n} days! Invest {amt} in our crypto scheme and double your money. {url}"),
    ("investment", "english", "social", "Join our VIP trading group. Our expert gives sure-shot tips with guaranteed profit. Deposit {amt} to start today. {url}"),
    ("investment", "english", "email", "Exclusive pre-launch offer: risk-free investment with fixed daily income. Limited slots, invest {amt} now."),
    ("investment", "english", "sms", "Earn passive income of {amt} every day with our AI trading bot. 100% profit guaranteed. Register: {url}"),
    # ---- prize_lottery
    ("prize_lottery", "english", "sms", "Congratulations! You have won a lucky draw prize of Rs. 25,00,000. Pay {amt} claim fee to receive your reward. {url}"),
    ("prize_lottery", "english", "email", "You are our lucky winner of a free smartphone. Confirm your address and pay the delivery charge of {amt} today only."),
    ("prize_lottery", "english", "sms", "WINNER! Your number was selected in the KBC lottery. Contact our agent and pay {amt} tax to release the prize."),
    ("prize_lottery", "tamil_english", "sms", "Congratulations! Neenga lucky draw-la prize jeythirukkeenga. Claim panna {amt} fee anuppunga: {url}"),
    # ---- support_impersonation
    ("support_impersonation", "english", "sms", "This is {bank} customer care. Your account has suspicious activity. Install AnyDesk and share the code so we can secure it."),
    ("support_impersonation", "english", "email", "Amazon Support: your order of {amt} will be charged. If this was not you, call our helpline immediately and share the OTP."),
    ("support_impersonation", "english", "sms", "Your courier is held at the depot. Pay a re-delivery fee of {amt} at {url} to avoid return."),
    ("support_impersonation", "english", "social", "Hello, I am from technical support. Your device has a virus. Download the support app at {url} and give us access."),
    # ---- govt_impersonation
    ("govt_impersonation", "english", "sms", "Income Tax Department: refund of {amt} approved. Submit your bank details at {url} within {n} hours."),
    ("govt_impersonation", "english", "sms", "Cyber Police Notice: a case is registered against your Aadhaar. Pay a penalty of {amt} immediately or face arrest."),
    ("govt_impersonation", "english", "email", "Customs Department: your parcel contains illegal items. Pay {amt} clearance fee at {url} to avoid legal action."),
    ("govt_impersonation", "english", "sms", "TRAI: your mobile number will be disconnected today. Press 1 to speak to an officer and verify your Aadhaar."),
    # ---- account_verification
    ("account_verification", "english", "email", "Security notice: unusual sign-in detected. Verify your account within {n} hours or it will be closed. {url}"),
    ("account_verification", "english", "email", "Dear user, your mailbox is almost full and will be deactivated. Kindly verify your account here: {url}"),
    ("account_verification", "english", "sms", "Your account has been temporarily limited. Confirm your identity now to restore access: {url}"),
    ("account_verification", "english", "social", "We noticed a policy violation on your page. Verify your account in {n} hours or it will be permanently disabled. {url}"),
    # ---- credential_theft
    ("credential_theft", "english", "email", "Your password expires today. Log in at {url} and enter your current password to keep your account active."),
    ("credential_theft", "english", "sms", "To secure your account, reply with your OTP and card number. Do it immediately."),
    ("credential_theft", "english", "email", "IT Helpdesk: confirm your username and password at {url} to complete the mandatory security upgrade."),
    ("credential_theft", "english", "sms", "{bank}: your card is locked. Share your CVV and OTP to unlock it right now."),
    # ---- phishing (generic)
    ("phishing", "english", "email", "Please review the attached invoice and sign in to view the document: {url}"),
    ("phishing", "english", "email", "A shared file is waiting for you. Kindly click the link below to open it. {url}"),
    ("phishing", "english", "sms", "Package could not be delivered. Click {url} to reschedule your delivery now."),
    ("phishing", "english", "email", "Your subscription payment was declined. Update your billing information immediately at {url} to avoid cancellation."),
    # ---- legitimate (includes hard negatives containing urgent/bank/OTP words)
    ("legitimate", "english", "sms", "Your OTP is 482913. It is valid for 10 minutes. Do not share this OTP with anyone, including bank staff."),
    ("legitimate", "english", "sms", "{bank}: Rs. 1,250 debited from A/c XX4521 on 03-Oct. If not you, call the number on the back of your card."),
    ("legitimate", "english", "sms", "Your order has been shipped and will arrive by Friday. Track it in the app. Thank you for shopping with us."),
    ("legitimate", "english", "email", "Hi {name}, the team meeting is moved to 3 PM tomorrow. Please update your calendar. Regards, Priya"),
    ("legitimate", "english", "email", "Your electricity bill of {amt} for September is generated. You can pay it in the official app before the due date."),
    ("legitimate", "english", "sms", "Reminder: your appointment is scheduled for Monday at 10 AM. Reply C to confirm or call the clinic to reschedule."),
    ("legitimate", "english", "social", "Hey {name}, are we still on for dinner on Saturday? Let me know what time works for you."),
    ("legitimate", "english", "email", "Dear {name}, thank you for attending the interview at {co}. We will share the result via the careers portal within 7 days. No fees are charged at any stage."),
    ("legitimate", "english", "sms", "Your credit card statement is ready. Never share your card details or OTP with anyone. Log in to the official app to view it."),
    ("legitimate", "english", "email", "Assignment 3 is due on Friday. Please upload your submission on the college portal. Contact the faculty if you need an extension."),
    ("legitimate", "english", "sms", "Your train PNR 4521897 is confirmed. Coach S4, seat 23. Carry a valid ID proof during the journey."),
    ("legitimate", "english", "email", "Your password was changed successfully. If you did not make this change, open the official website directly and reset it."),
    ("legitimate", "english", "social", "Congrats {name} on your new job! Let's catch up this weekend and celebrate."),
    ("legitimate", "english", "sms", "Your {bank} credit of {amt} was received. Available balance updated. Thank you for banking with us."),
    ("legitimate", "tamil_english", "sms", "Hi {name}, naalaiku meeting 10 AM-ku irukku. Please time-ku vanga. Thanks."),
    ("legitimate", "tamil_english", "social", "Da, inniku evening cricket match irukku, vara mudiyuma? Let me know."),
    ("legitimate", "tamil_english", "sms", "Ungal order dispatch aagiduchu. Naalaiku delivery varum. Thank you."),
    ("legitimate", "english", "sms", "Your {bank} OTP for the online purchase of {amt} is 719364. Do not share it with anyone. If you did not request it, ignore this message."),
    ("legitimate", "english", "email", "Hi {name}, your offer letter from {co} is attached. Please sign and return it by Friday. Reach out to HR if you have questions. No payment is required."),
    ("legitimate", "english", "sms", "Your mobile recharge of {amt} was successful. Validity extended by 28 days. Thank you."),
    ("legitimate", "english", "email", "Reminder: the library book you borrowed is due tomorrow. Please return it to avoid a late fee as per college rules."),
    ("legitimate", "english", "sms", "Your cab is arriving in 3 minutes. Driver: Ramesh, vehicle TN 09 AB 1234. Share the trip OTP 5521 only with the driver."),
    ("legitimate", "english", "email", "Your payment of {amt} to {co} was received. A receipt is attached. This is an automated message, no action needed."),
    ("legitimate", "english", "social", "Hey, urgent: the project report needs to be submitted today by 5 PM. Can you send me your part?"),
    ("legitimate", "english", "email", "Security alert: a new sign-in to your account from Chennai. If this was you, no action is needed. If not, change your password from the official app."),
    ("legitimate", "english", "sms", "Dear {name}, your {bank} fixed deposit matures on 15-Nov. Visit your nearest branch or the official app to renew."),
    ("legitimate", "tamil_english", "sms", "Ungal {bank} statement ready aagiduchu. Official app-la login pannitu paarunga. OTP yaarukum share pannathinga."),
    ("legitimate", "tamil_english", "social", "Machan, project submission naalaiku, unga part anuppunga please. Thanks da."),
    ("legitimate", "tamil_english", "sms", "Ungal appointment Monday 10 AM-ku confirm aagiduchu. Reschedule panna clinic-ku call pannunga."),
]


def fill(tpl: str, rng: random.Random) -> str:
    out = tpl
    for k, vals in SLOTS.items():
        token = "{" + k + "}"
        while token in out:
            out = out.replace(token, rng.choice(vals), 1)
    return out


def assign_splits(groups, rng):
    """Group-level split per category (~60/20/20). groups: list of (gid, category)."""
    by_cat = {}
    for gid, cat in groups:
        by_cat.setdefault(cat, []).append(gid)
    split = {}
    for cat, ids in by_cat.items():
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.2))
        for j, gid in enumerate(ids):
            split[gid] = "test" if j < n_test else "val" if j < n_test + n_val else "train"
    return split


def build() -> pd.DataFrame:
    from ml.cores import CORES, OPENERS, TAILS, LINKS_SCAM, LINKS_LEGIT
    rng = random.Random(SEED)
    groups = [(i, t[0]) for i, t in enumerate(T)]
    core_list = []                      # (gid, cat, lang, core)
    gid = 1000
    for cat, items in CORES.items():
        for lang, core in items:
            core_list.append((gid, cat, lang, core)); groups.append((gid, cat)); gid += 1
    split = assign_splits(groups, rng)
    # Make sure the code-mixed (Tamil-English) test subset contains SCAM examples too:
    # alternate Tamil scam cores between test and train so both sides see some.
    tamil_scam = sorted([(c, g) for g, c, l, _ in core_list if l == "tamil_english" and c != "legitimate"])
    for k, (_, g) in enumerate(tamil_scam):
        split[g] = "test" if k % 2 == 0 else "train"
    rows = []

    def add(txt, cat, lang, src, g):
        rows.append({"text": txt, "language": lang, "category": cat,
                     "label": "legitimate" if cat == "legitimate" else "scam",
                     "source_type": src, "template_id": g, "split": split[g], "data_origin": "synthetic"})

    for gi, (cat, lang, src, tpl) in enumerate(T):
        seen = set()
        for _ in range(VARIANTS_PER_TEMPLATE * 3):
            txt = fill(tpl, rng)
            if txt not in seen:
                seen.add(txt); add(txt, cat, lang, src, gi)
            if len(seen) >= VARIANTS_PER_TEMPLATE:
                break
    for g, cat, lang, core in core_list:
        seen = set()
        for _ in range(40):
            links = LINKS_LEGIT if cat == "legitimate" else LINKS_SCAM
            body = rng.choice(OPENERS) + core[0].upper() + core[1:] + "." + rng.choice(TAILS) + rng.choice(links)
            txt = fill(body, rng)
            if txt not in seen:
                seen.add(txt); add(txt, cat, lang, rng.choice(["sms", "email", "social"]), g)
            if len(seen) >= 6:
                break
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_csv(OUT / "scam_dataset.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} synthetic rows to {OUT / 'scam_dataset.csv'}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(df.groupby("category").size())
