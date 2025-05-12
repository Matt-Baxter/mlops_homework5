from src.retriever.retriever import get_similar_responses

def test_get_similar_responses():
    """Test that the retriever returns results"""
    # We can't test exact responses since they depend on the embeddings
    # Instead, test that we get the expected format and type
    results = get_similar_responses("What is quantum computing?")
    
    # Check if we got a list of strings
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(item, str) for item in results)
    
    # Verify the results are not empty
    assert all(len(item) > 0 for item in results)