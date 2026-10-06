"""Run:  python -m unittest discover -s tests -v    (or: pytest -v)"""
import unittest
from app.services.analyzer import analyze, InputError
from app.services.url_analyzer import analyze_url
from app.rules.indicators import detect_indicators
from app.utils.upload import validate_upload, UploadError

HIGHISH = ("HIGH", "MEDIUM")


class TestLegitimate(unittest.TestCase):
    def test_otp_notice_not_flagged(self):
        r = analyze("Your OTP is 482913. Do not share this OTP with anyone, including bank staff.")
        self.assertEqual(r["risk"]["level"], "LOW"); self.assertEqual(r["indicators"], [])

    def test_bank_debit_alert(self):
        self.assertEqual(analyze("HDFC Bank: Rs. 1,250 debited from A/c XX4521. If not you, call the number on your card.")["risk"]["level"], "LOW")

    def test_casual_chat(self):
        self.assertEqual(analyze("Hey, are we still meeting at the library tomorrow at 4?")["risk"]["level"], "LOW")

    def test_legit_with_urgent_word(self):
        r = analyze("Hey, urgent: the project report needs to be submitted today by 5 PM. Can you send me your part?")
        self.assertNotEqual(r["risk"]["level"], "HIGH")


class TestScams(unittest.TestCase):
    def test_obvious_phishing(self):
        r = analyze("Dear customer, your SBI account will be blocked today. Verify immediately: http://sbi-kyc-update.example.com/verify")
        self.assertEqual(r["risk"]["level"], "HIGH")
        self.assertTrue(r["urls"] and r["urls"][0]["risk_score"] >= 0.3)

    def test_subtle_phishing(self):
        r = analyze("A shared file is waiting for you. Kindly open the link to view it: https://docs-share.example.com/view")
        self.assertIn(r["risk"]["level"], HIGHISH + ("UNCERTAIN",))

    def test_ai_style_scam(self):
        t = ("Dear valued customer, we regret to inform you that unusual activity has been detected. "
             "Kindly verify your account at http://secure-login.example.net/auth to avoid suspension.")
        r = analyze(t)
        self.assertIn(r["risk"]["level"], HIGHISH)
        self.assertIn(r["ai_content"]["level"], ("weak", "possible"))
        self.assertNotIn("definitely", r["ai_content"]["label"].lower())

    def test_tamil_english(self):
        r = analyze("Your bank account block aagidum, immediately verify pannunga.")
        self.assertEqual(r["language"], "tamil_english"); self.assertIn(r["risk"]["level"], HIGHISH)

    def test_job_scam(self):
        r = analyze("Congratulations! You are selected for work from home at TechNova. Pay a refundable registration fee of Rs. 999 to confirm.")
        self.assertIn(r["risk"]["level"], HIGHISH)

    def test_upi_scam(self):
        r = analyze("You have a collect request of Rs. 5,000. Approve it and enter your UPI PIN to receive the refund.")
        self.assertIn(r["risk"]["level"], HIGHISH)
        self.assertIn("Sensitive information request", {i["indicator"] for i in r["indicators"]})

    def test_kyc_scam(self):
        r = analyze("Your HDFC Bank KYC has expired. Update PAN immediately or your account will be suspended. https://bit.ly/3xKycUp")
        self.assertIn(r["risk"]["level"], HIGHISH)


class TestUrls(unittest.TestCase):
    def test_ip_url(self):
        self.assertIn("IP-based URL", [i["indicator"] for i in analyze_url("http://192.0.2.15/login")["indicators"]])

    def test_shortener(self):
        self.assertIn("URL shortener", [i["indicator"] for i in analyze_url("https://bit.ly/abc")["indicators"]])

    def test_lookalike(self):
        self.assertIn("Look-alike / brand misuse", [i["indicator"] for i in analyze_url("https://hdfc-secure-login.example.com")["indicators"]])

    def test_official_domain_not_lookalike(self):
        self.assertNotIn("Look-alike / brand misuse", [i["indicator"] for i in analyze_url("https://www.sbi.co.in/")["indicators"]])

    def test_url_never_fetched(self):
        self.assertFalse(analyze_url("http://192.0.2.15/x")["fetched"])

    def test_url_only_input(self):
        r = analyze("http://192.0.2.15/login")
        self.assertEqual(r["input_type"], "url"); self.assertIn(r["risk"]["level"], HIGHISH)


class TestInvalid(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(InputError): analyze("   ")

    def test_none(self):
        with self.assertRaises(InputError): analyze(None)

    def test_wrong_type(self):
        with self.assertRaises(InputError): analyze(12345)

    def test_very_long(self):
        with self.assertRaises(InputError): analyze("a" * 20000)

    def test_control_chars_stripped(self):
        self.assertEqual(analyze("Hello\x00 there, see you at 5\x07 pm")["input_type"], "text")

    def test_html_input_is_plain_text(self):
        r = analyze("<script>alert(1)</script> your account will be blocked, verify immediately")
        # Input is treated purely as text; the engine must not crash and must still find the scam wording.
        self.assertIn("Urgency", {i["indicator"] for i in r["indicators"]})

    def test_unsupported_file(self):
        with self.assertRaises(UploadError): validate_upload("malware.exe", 100)

    def test_oversized_file(self):
        with self.assertRaises(UploadError): validate_upload("a.png", 10 * 1024 * 1024)

    def test_ocr_not_enabled_message(self):
        with self.assertRaises(UploadError) as c: validate_upload("shot.png", 1000)
        self.assertIn("OCR", str(c.exception))


class TestEvidence(unittest.TestCase):
    def test_structured_evidence(self):
        ev = detect_indicators("Your account will be blocked. Verify immediately.")
        for e in ev:
            for k in ("indicator", "severity", "evidence", "explanation"): self.assertIn(k, e)

    def test_negation_suppresses_otp_request(self):
        self.assertEqual([e for e in detect_indicators("Never share your OTP with anyone.") if e["indicator"] == "Sensitive information request"], [])


if __name__ == "__main__":
    unittest.main()
