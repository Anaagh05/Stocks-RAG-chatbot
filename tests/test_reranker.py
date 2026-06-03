import pytest
from unittest.mock import MagicMock
import numpy as np

from src.retriever import RetrievedChunk
from src.reranker import CrossEncoderReranker
from src.confidence import ConfidenceThresholder, ConfidenceTier

@pytest.fixture
def mock_cross_encoder(mocker):
    # Mock the sentence_transformers.CrossEncoder
    mock = mocker.patch("src.reranker.CrossEncoder")
    mock_instance = MagicMock()
    mock.return_value = mock_instance
    return mock_instance

def test_reranker_sorting(mock_cross_encoder):
    # Arrange
    reranker = CrossEncoderReranker()
    
    candidates = [
        RetrievedChunk(chunk_id="1", text="Unrelated text", metadata={}),
        RetrievedChunk(chunk_id="2", text="Highly relevant text", metadata={}),
        RetrievedChunk(chunk_id="3", text="Somewhat relevant text", metadata={}),
    ]
    
    # Predict returns scores for each pair in order.
    # We mock it to return a low score for 1, high for 2, medium for 3.
    mock_cross_encoder.predict.return_value = np.array([-2.0, 5.5, 1.2])
    
    # Act
    results = reranker.rerank("query", candidates, top_k=2)
    
    # Assert
    assert len(results) == 2
    assert results[0].chunk_id == "2"
    assert results[0].cross_encoder_score == 5.5
    assert results[1].chunk_id == "3"
    assert results[1].cross_encoder_score == 1.2

def test_confidence_evaluation():
    thresholder = ConfidenceThresholder()
    
    assert thresholder.classify(0.95) == ConfidenceTier.HIGH
    assert thresholder.classify(0.60) == ConfidenceTier.MODERATE
    assert thresholder.classify(0.20) == ConfidenceTier.LOW

def test_reranker_empty_candidates(mock_cross_encoder):
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", [])
    assert len(results) == 0
