from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_query_endpoint():
    """Test that the query endpoint returns results"""
    response = client.post("/similar_responses", json={"question": "What is quantum computing?"})
    
    # Check status code
    assert response.status_code == 200
    
    # Check response format
    response_data = response.json()
    assert "answers" in response_data
    assert isinstance(response_data["answers"], list)
    
    # Check that we got results
    assert len(response_data["answers"]) > 0
    assert all(isinstance(item, str) for item in response_data["answers"])