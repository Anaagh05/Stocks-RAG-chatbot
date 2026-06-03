import pytest
from src.retriever import HybridRetriever, RRF_K

def test_rrf_merge():
    # Format: (chunk_id, text, metadata)
    dense_results = [
        ("chunk_A", "Text A", {"source": "dense"}),  # Rank 1
        ("chunk_B", "Text B", {"source": "dense"}),  # Rank 2
        ("chunk_C", "Text C", {"source": "dense"}),  # Rank 3
    ]
    
    bm25_results = [
        ("chunk_B", "Text B", {"source": "bm25"}),   # Rank 1
        ("chunk_D", "Text D", {"source": "bm25"}),   # Rank 2
        ("chunk_A", "Text A", {"source": "bm25"}),   # Rank 3
    ]
    
    fused = HybridRetriever._rrf_merge(dense_results, bm25_results, top_k=5)
    
    assert len(fused) == 4
    
    # Calculate expected scores
    # chunk_B: dense rank 2, bm25 rank 1 -> 1/(60+2) + 1/(60+1)
    # chunk_A: dense rank 1, bm25 rank 3 -> 1/(60+1) + 1/(60+3)
    # chunk_C: dense rank 3 -> 1/(60+3)
    # chunk_D: bm25 rank 2 -> 1/(60+2)
    score_B = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
    score_A = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 3)
    score_D = 1.0 / (RRF_K + 2)
    score_C = 1.0 / (RRF_K + 3)
    
    assert fused[0].chunk_id == "chunk_B"
    assert fused[1].chunk_id == "chunk_A"
    assert fused[2].chunk_id == "chunk_D"
    assert fused[3].chunk_id == "chunk_C"
    
    assert abs(fused[0].rrf_score - score_B) < 1e-6
    assert abs(fused[1].rrf_score - score_A) < 1e-6

def test_rrf_merge_empty_lists():
    fused = HybridRetriever._rrf_merge([], [], top_k=5)
    assert len(fused) == 0

def test_rrf_merge_one_sided():
    dense_results = [("chunk_X", "Text X", {})]
    fused = HybridRetriever._rrf_merge(dense_results, [], top_k=5)
    assert len(fused) == 1
    assert fused[0].chunk_id == "chunk_X"
    assert fused[0].dense_rank == 1
    assert fused[0].bm25_rank is None
