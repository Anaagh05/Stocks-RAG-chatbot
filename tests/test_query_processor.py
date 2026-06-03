import pytest
from src.query_processor import QueryProcessor, ADVISORY_REFUSAL

@pytest.fixture
def processor():
    return QueryProcessor()

def test_advisory_intent_detection(processor):
    advisory_queries = [
        "Should I invest in SBI Small Cap?",
        "Which is better, SBI Bluechip or HDFC?",
        "Recommend a good SIP.",
        "What will be the return in 5 years?",
        "Is SBI Contra worth investing?",
        "Please advise on where to invest."
    ]
    
    for query in advisory_queries:
        result = processor.process(query)
        assert result.is_advisory is True, f"Failed to catch advisory query: {query}"
        assert result.refusal_message == ADVISORY_REFUSAL

def test_factual_intent_allowed(processor):
    factual_queries = [
        "What is the exit load of SBI Small Cap Fund?",
        "Who is the fund manager for SBI Contra Fund?",
        "What are the top 10 holdings of SBI Bluechip?",
        "Give me the investment objective of SBI Nifty Index Fund."
    ]
    
    for query in factual_queries:
        result = processor.process(query)
        assert result.is_advisory is False, f"Falsely flagged factual query: {query}"

def test_fund_name_normalization(processor):
    # Test informal names map to canonical names
    cases = [
        ("What is the NAV of sbi bluechip?", "SBI Bluechip Fund"),
        ("Tell me about sbi small cap.", "SBI Small Cap Fund"),
        ("sbi tech fund details", "SBI Technology Opportunities Fund"),
        ("sbi midcap exit load", "SBI Magnum Midcap Fund")
    ]
    
    for query, expected_fund in cases:
        result = processor.process(query)
        assert result.detected_fund == expected_fund
        assert result.chroma_filter == {"fund_name": expected_fund}
        assert expected_fund in result.cleaned_query

def test_no_fund_detected(processor):
    query = "What is a mutual fund?"
    result = processor.process(query)
    assert result.detected_fund is None
    assert result.chroma_filter is None
