"""
Test script to verify the chatbot can answer questions about specific topics
"""
import os
import sys
import asyncio
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Create a local Qdrant client for testing
def _generate_embedding(text: str) -> list[float]:
    """Generate embedding for text using a simple hashing method"""
    # Fallback: Use a simple method to generate embeddings
    # This is a very basic method that just hashes the text and converts to floats
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    embedding = []
    for i in range(0, len(text_hash), 2):
        if i + 1 < len(text_hash):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to [0, 1]
            embedding.append(value)

    # Pad or truncate to 1536 dimensions (OpenAI embedding size)
    while len(embedding) < 1536:
        embedding.append(0.0)

    return embedding[:1536]

# Create a persistent Qdrant client that matches the indexing script
import os
storage_path = os.path.join(os.path.dirname(__file__), "qdrant_storage")
test_client = QdrantClient(path=storage_path)

# Collection name
collection_name = "textbook_content"

async def test_rag_search(query: str, limit: int = 5):
    """Test RAG functionality with the in-memory client"""
    query_embedding = _generate_embedding(query)

    search_results = test_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=limit
    )

    results = []
    for result in search_results:
        results.append({
            "content_id": result.payload["content_id"],
            "text": result.payload["text"],
            "score": result.score,
            "metadata": result.payload["metadata"]
        })

    return results

def test_rag_functionality():
    """Test the RAG system directly"""
    print("Testing RAG functionality...")

    # Test query about a specific topic
    test_queries = [
        "What is ROS2?",
        "How does Gazebo work?",
        "Explain NVIDIA Isaac platform",
        "What are Vision Language Action models?",
        "Humanoid robotics applications"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            # Test the in-memory RAG search
            results = asyncio.run(test_rag_search(query, limit=3))

            if results:
                print(f"Found {len(results)} results:")
                for i, result in enumerate(results[:2]):  # Show top 2 results
                    print(f"  Result {i+1}:")
                    print(f"    Text preview: {result['text'][:200]}...")
                    print(f"    Source: {result.get('metadata', {}).get('source', 'N/A')}")
                    print(f"    Score: {result.get('score', 'N/A')}")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  Error during search: {e}")

def test_chatbot_response():
    """Test the chatbot response functionality"""
    print("\n" + "="*50)
    print("Testing Chatbot Response Functionality")
    print("="*50)

    # Test questions about specific topics
    test_questions = [
        "What is ROS2?",
        "Explain how Gazebo simulation works",
        "What is the NVIDIA Isaac platform used for?",
        "Tell me about Vision Language Action models",
        "What are the main challenges in humanoid robotics?"
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            # Test the core logic of the chat function
            print("  Searching for relevant content...")
            search_results = asyncio.run(test_rag_search(question, limit=3))
            print(f"  Found {len(search_results)} relevant results")

            if search_results:
                context_text = "\n".join([result['text'] for result in search_results[:3]])  # Use top 3 results
                print(f"  Context preview: {context_text[:300]}...")
            else:
                print("  No relevant context found in documentation")

        except Exception as e:
            print(f"  Error during chatbot test: {e}")

def check_qdrant_contents():
    """Check what's in the Qdrant collection"""
    print("\n" + "="*50)
    print("Checking Qdrant Collection Contents")
    print("="*50)

    try:
        collection_info = test_client.get_collection(collection_name)
        print(f"Collection: {collection_name}")
        print(f"Vectors count: {collection_info.points_count}")

        # Try to get a few sample points
        if collection_info.points_count > 0:
            # Get some sample points
            points = test_client.scroll(
                collection_name=collection_name,
                limit=3
            )[0]  # scroll returns (points, next_page_offset)

            print(f"\nSample points from the collection:")
            for i, point in enumerate(points):
                payload = point.payload
                print(f"  Point {i+1}:")
                print(f"    Source: {payload.get('metadata', {}).get('source', 'Unknown')}")
                print(f"    Title: {payload.get('metadata', {}).get('title', 'Unknown')}")
                print(f"    Text preview: {payload['text'][:100]}...")
    except Exception as e:
        print(f"Error checking Qdrant contents: {e}")

if __name__ == "__main__":
    print("Running chatbot functionality tests...")

    # First check what's in Qdrant
    check_qdrant_contents()

    # Test RAG functionality
    test_rag_functionality()

    # Test chatbot response
    test_chatbot_response()

    print("\n" + "="*50)
    print("Tests completed!")
    print("If the RAG system returns relevant results, the chatbot should now answer correctly based on the documentation.")
    print("="*50)