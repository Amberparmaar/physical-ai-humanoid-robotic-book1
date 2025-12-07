import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch, AsyncMock

client = TestClient(app)

@pytest.fixture
def mock_rag_service():
    with patch('api.routes.rag.rag_service') as mock_service:
        yield mock_service

@pytest.fixture
def mock_embedding_service():
    with patch('api.routes.rag.embedding_service') as mock_service:
        yield mock_service

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Physical AI & Humanoid Robotics RAG API"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_create_embeddings_success(mock_embedding_service):
    mock_embedding_service.create_embeddings = AsyncMock(return_value="vector-123")
    
    content_data = {
        "content_id": "ch1-intro",
        "module": "ROS2",
        "section": "Introduction",
        "page_number": 1,
        "topic": "overview",
        "difficulty_level": "beginner",
        "content_type": "text",
        "language": "en",
        "content_text": "ROS2 is the next generation of Robot Operating System..."
    }
    
    response = client.post("/api/embeddings", json=content_data)
    
    assert response.status_code == 200
    assert response.json() == {"vector_id": "vector-123"}

def test_query_rag_success(mock_rag_service):
    mock_rag_service.query = AsyncMock(return_value={
        "results": [],
        "sources": ["ch1-1"]
    })
    
    response = client.post("/api/query", params={"query": "What is ROS2?", "top_k": 3, "user_id": "test-user"})
    
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data