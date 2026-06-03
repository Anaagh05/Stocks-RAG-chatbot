import pytest
from src.guardrail import PIIScrubber

@pytest.fixture
def scrubber():
    return PIIScrubber()

def test_pan_card_redaction(scrubber):
    query = "My PAN is ABCDE1234F, is it safe?"
    cleaned, redactions = scrubber.scrub(query)
    assert "[REDACTED_PAN]" in cleaned
    assert "ABCDE1234F" not in cleaned
    assert "[REDACTED_PAN]" in redactions

def test_aadhaar_redaction(scrubber):
    query = "My Aadhaar is 1234 5678 9012."
    cleaned, redactions = scrubber.scrub(query)
    assert "[REDACTED_AADHAAR]" in cleaned
    assert "1234 5678 9012" not in cleaned
    assert "[REDACTED_AADHAAR]" in redactions

def test_phone_number_redaction(scrubber):
    query = "Call me at +91 9876543210 for details."
    cleaned, redactions = scrubber.scrub(query)
    assert "[REDACTED_PHONE]" in cleaned
    assert "9876543210" not in cleaned
    assert "[REDACTED_PHONE]" in redactions

def test_email_redaction(scrubber):
    query = "Email me at test@example.com."
    cleaned, redactions = scrubber.scrub(query)
    assert "[REDACTED_EMAIL]" in cleaned
    assert "test@example.com" not in cleaned
    assert "[REDACTED_EMAIL]" in redactions

def test_otp_redaction(scrubber):
    query = "My OTP is 123456"
    cleaned, redactions = scrubber.scrub(query)
    assert "[REDACTED_OTP]" in cleaned
    assert "123456" not in cleaned
    assert "[REDACTED_OTP]" in redactions

def test_account_number_redaction(scrubber):
    query = "Account number 12345678901234"
    cleaned, redactions = scrubber.scrub(query)
    assert "[REDACTED_ACCOUNT]" in cleaned
    assert "12345678901234" not in cleaned
    assert "[REDACTED_ACCOUNT]" in redactions

def test_no_pii(scrubber):
    query = "What is the exit load for SBI Small Cap Fund?"
    cleaned, redactions = scrubber.scrub(query)
    assert cleaned == query
    assert len(redactions) == 0

def test_contains_pii_helper(scrubber):
    assert scrubber.contains_pii("PAN: ABCDE1234F") is True
    assert scrubber.contains_pii("Hello world") is False
