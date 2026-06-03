import pytest
from unittest.mock import MagicMock
from src.retriever import RetrievedChunk
from src.reranker import CrossEncoderReranker
from src.confidence import ConfidenceThresholder, ConfidenceTier

@pytest.fixture
def mock_cohere(mocker):
    # Mock the cohere.Client
    mock_class = mocker.patch("src.reranker.cohere.Client")
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    return mock_instance

def test_reranker_sorting(mock_cohere):
    # Arrange
    reranker = CrossEncoderReranker()
    
    candidates = [
        RetrievedChunk(chunk_id="1", text="Unrelated text", metadata={}),
        RetrievedChunk(chunk_id="2", text="Highly relevant text", metadata={}),
        RetrievedChunk(chunk_id="3", text="Somewhat relevant text", metadata={}),
    ]
    
    # Mock Cohere response
    mock_result_1 = MagicMock()
    mock_result_1.index = 1
    mock_result_1.relevance_score = 0.95
    
    mock_result_2 = MagicMock()
    mock_result_2.index = 2
    mock_result_2.relevance_score = 0.60
    
    mock_response = MagicMock()
    mock_response.results = [mock_result_1, mock_result_2]
    
    mock_cohere.rerank.return_value = mock_response
    
    # Act
    results = reranker.rerank("query", candidates, top_k=2)
    
    # Assert
    assert len(results) == 2
    assert results[0].chunk_id == "2"
    assert results[0].cross_encoder_score == 0.95
    assert results[1].chunk_id == "3"
    assert results[1].cross_encoder_score == 0.60

def test_confidence_evaluation():
    thresholder = ConfidenceThresholder()
    
    assert thresholder.classify(0.95) == ConfidenceTier.HIGH
    assert thresholder.classify(0.60) == ConfidenceTier.MODERATE
    assert thresholder.classify(0.20) == ConfidenceTier.LOW

def test_reranker_empty_candidates(mock_cohere):
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", [])
    assert len(results) == 0
