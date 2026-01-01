"""
Script to index documentation content into the Qdrant vector store
This script reads markdown files from the docs directory and adds them to the RAG system
"""
import os
import sys
import hashlib
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.rag import rag_service
from config import settings
import openai
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def index_documentation(docs_path):
    """Index documentation content into the RAG service"""
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

        # Add each chunk to the RAG service
        for chunk_idx, chunk in enumerate(chunks):
            # Create a unique ID for this chunk
            content_id = abs(hash(f"{file_info['path']}_chunk_{chunk_idx}")) % (10 ** 9)  # Limit to 9 digits

            # Add to RAG service (the rag_service handles embedding generation internally)
            try:
                rag_service.add_content(
                    content_id=content_id,
                    text=chunk,
                    metadata={
                        'source': str(file_info['path']),
                        'title': file_info['filename'],
                        'chunk_id': chunk_idx
                    }
                )
                logger.info(f"Added chunk {chunk_idx} from {file_info['path']} to RAG service")
                total_chunks += 1
            except Exception as e:
                logger.error(f"Error adding content to RAG service: {e}")

    logger.info(f"Successfully indexed {len(files)} documentation files with {total_chunks} total chunks")

if __name__ == "__main__":
    docs_path = "../../../docs"  # Relative path from this script's location

    # Verify Qdrant connection before starting
    try:
        logger.info("Checking Qdrant connection...")
        # This will trigger initialization of the Qdrant client
        collection_info = rag_service.client.get_collection(rag_service.collection_name)
        logger.info(f"Qdrant connection successful. Collection '{rag_service.collection_name}' exists.")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        logger.info("Make sure Qdrant is running on the configured host and port.")
        sys.exit(1)

    index_documentation(docs_path)