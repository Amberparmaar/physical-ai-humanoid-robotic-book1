from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Physical AI & Humanoid Robotics RAG API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for responses
mock_responses = {
    "hello": "Hello! I'm your AI tutor for Physical AI & Humanoid Robotics. How can I help you?",
    "ros2": "ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides services such as hardware abstraction, device drivers, libraries, and more.",
    "gazebo": "Gazebo is a robot simulation environment that allows you to create realistic robot models and test them in a safe virtual space.",
    "isaac": "The NVIDIA Isaac platform provides tools and libraries for developing GPU-accelerated robotic applications.",
    "vla": "Vision Language Action (VLA) models combine visual perception, language understanding, and action planning for humanoid robots."
}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Physical AI & Humanoid Robotics RAG API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/query")
async def query_rag(query: str, top_k: int = 5, user_id: str = None):
    # Simple mock response - in a real implementation, this would connect to your RAG system
    query_lower = query.lower()
    response_text = "I found relevant information for your query. Please refer to the appropriate course materials."
    
    for keyword, response in mock_responses.items():
        if keyword in query_lower:
            response_text = response
            break
    
    return {
        "results": [
            {
                "content_id": "mock_content_1",
                "module": "mock_module",
                "content": response_text,
                "score": 0.9
            }
        ],
        "sources": ["mock_source_1"]
    }

@app.post("/api/embeddings")
async def create_embeddings(content):
    # Mock endpoint - in a real implementation, this would create embeddings
    return {"vector_id": "mock_vector_id_123"}