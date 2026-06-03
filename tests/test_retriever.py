import pytest
from unittest.mock import MagicMock
from src.retriever import Retriever, RetrievedChunk

@pytest.fixture
def mock_pinecone_and_embedder(mocker):
    # Mock Embedder
    mock_embedder_class = mocker.patch("src.retriever.Embedder")
    mock_embedder_inst = MagicMock()
    mock_embedder_inst.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embedder_class.return_value = mock_embedder_inst
    
    # Mock Pinecone
    mock_pinecone_class = mocker.patch("src.retriever.Pinecone")
    mock_pinecone_inst = MagicMock()
    mock_index = MagicMock()
    
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "chunk_A",
                "score": 0.95,
                "metadata": {"text": "Text A", "source": "test"}
            },
            {
                "id": "chunk_B",
                "score": 0.85,
                "metadata": {"text": "Text B", "source": "test"}
            }
        ]
    }
    
    mock_pinecone_inst.Index.return_value = mock_index
    mock_pinecone_class.return_value = mock_pinecone_inst
    
    return mock_embedder_inst, mock_index

def test_pinecone_retrieval(mock_pinecone_and_embedder):
    _, mock_index = mock_pinecone_and_embedder
    
    retriever = Retriever()
    results = retriever.retrieve("test query", top_k=2)
    
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_A"
    assert results[0].text == "Text A"
    assert results[0].pinecone_score == 0.95
    assert "text" not in results[0].metadata  # Cleaned out
    
    mock_index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=2,
        include_values=False,
        include_metadata=True
    )
