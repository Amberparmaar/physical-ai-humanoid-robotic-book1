"""
Test script to run an in-memory Qdrant instance and index documentation
"""
import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams
from config import settings
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_local_qdrant_client():
    """Create a persistent Qdrant client for testing"""
    try:
        # Create a persistent instance of Qdrant
        # Using a local path to store the data
        import tempfile
        import os
        storage_path = os.path.join(os.path.dirname(__file__), "qdrant_storage")
        client = QdrantClient(path=storage_path)  # Persistent on-disk storage
        return client
    except Exception as e:
        logger.error(f"Error creating Qdrant client: {e}")
        # Fallback to in-memory if path fails
        try:
            client = QdrantClient(":memory:")
            logger.info("Using in-memory storage as fallback")
            return client
        except Exception as e2:
            logger.error(f"Both persistent and in-memory storage failed: {e2}")
            return None

def read_markdown_files(docs_path):
    """Read all markdown files from the documentation directory"""
    markdown_files = []

    docs_dir = Path(docs_path)

    # Walk through all directories and subdirectories
    for md_file in docs_dir.rglob("*.md"):
        if md_file.name != "intro.md":  # Skip intro.md as it's general
            relative_path = md_file.relative_to(docs_dir)
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                markdown_files.append({
                    'path': str(relative_path),
                    'content': content,
                    'filename': md_file.name
                })

    # Add intro.md at the end
    intro_path = docs_dir / "intro.md"
    if intro_path.exists():
        with open(intro_path, 'r', encoding='utf-8') as f:
            content = f.read()
            markdown_files.append({
                'path': "intro.md",
                'content': content,
                'filename': "intro.md"
            })

    return markdown_files

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

def index_documentation(docs_path):
    """Index documentation content into an in-memory Qdrant instance"""
    # Create local Qdrant client
    client = create_local_qdrant_client()
    if not client:
        logger.error("Failed to create Qdrant client")
        return False
    
    # Define collection name
    collection_name = "textbook_content"
    
    # Create collection
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Created collection '{collection_name}'")
    except Exception as e:
        logger.warning(f"Collection may already exist: {e}")
    
    logger.info(f"Reading documentation files from {docs_path}")
    files = read_markdown_files(docs_path)
    
    logger.info(f"Found {len(files)} markdown files")
    
    total_chunks = 0
    
    for i, file_info in enumerate(files):
        logger.info(f"Processing file: {file_info['path']}")
        
        # Split content into chunks if it's too large
        content = file_info['content']
        chunks = []
        chunk_size = 1000  # 1000 characters per chunk
        
        # Split by paragraphs first to avoid breaking up coherent sections
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If we still have very large paragraphs, split them further
        refined_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                # Split large chunks by sentences
                sentences = chunk.split('. ')
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk + sentence) < chunk_size:
                        temp_chunk += sentence + ". "
                    else:
                        if temp_chunk.strip():
                            refined_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + ". "
                if temp_chunk.strip():
                    refined_chunks.append(temp_chunk.strip())
            else:
                refined_chunks.append(chunk)
        
        chunks = refined_chunks
        
        # Add each chunk to the Qdrant collection
        for chunk_idx, chunk in enumerate(chunks):
            # Create a unique ID for this chunk
            content_id = abs(hash(f"{file_info['path']}_chunk_{chunk_idx}")) % (10 ** 9)  # Limit to 9 digits
            
            # Generate embedding for the text
            embedding = _generate_embedding(chunk)
            
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=content_id,
                            vector=embedding,
                            payload={
                                "content_id": content_id,
                                "text": chunk,
                                "metadata": {
                                    'source': str(file_info['path']),
                                    'title': file_info['filename'],
                                    'chunk_id': chunk_idx
                                }
                            }
                        )
                    ]
                )
                logger.info(f"Added chunk {chunk_idx} from {file_info['path']} to Qdrant")
                total_chunks += 1
            except Exception as e:
                logger.error(f"Error adding content to Qdrant: {e}")
    
    logger.info(f"Successfully indexed {len(files)} documentation files with {total_chunks} total chunks")
    
    # Save the client for potential later use
    # In a real application, you might want to save this to a persistent location
    logger.info("Indexing completed!")
    
    # Perform a simple test search to verify the indexing worked
    logger.info("\nPerforming a test search...")
    try:
        test_embedding = _generate_embedding("What is ROS2?")
        search_results = client.search(
            collection_name=collection_name,
            query_vector=test_embedding,
            limit=3
        )
        
        logger.info(f"Test search results for 'What is ROS2?':")
        for i, result in enumerate(search_results):
            logger.info(f"Result {i+1}: Score: {result.score}, Text preview: {result.payload['text'][:100]}...")
    except Exception as e:
        logger.error(f"Error during test search: {e}")
    
    return True

if __name__ == "__main__":
    docs_path = "../../../docs"  # Relative path from this script's location
    index_documentation(docs_path)