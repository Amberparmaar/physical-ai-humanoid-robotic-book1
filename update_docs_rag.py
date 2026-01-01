import os
import asyncio
import hashlib
from pathlib import Path
import logging

# Add the backend/src directory to the Python path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.rag import rag_service
from services.enhanced_rag import enhanced_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_markdown_files(docs_path: str) -> list:
    """
    Reads all markdown files from the docs directory and returns a list of content dictionaries.
    """
    content_list = []
    
    docs_dir = Path(docs_path)
    
    for md_file in docs_dir.rglob("*.md"):  # Recursively find all .md files
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create a unique ID based on the file path
            content_id = abs(hash(str(md_file))) % (10 ** 8)  # Keep ID within reasonable range
            
            # Extract directory and filename for metadata
            relative_path = md_file.relative_to(docs_dir)
            directory = relative_path.parent.name if relative_path.parent.name else 'root'
            filename = relative_path.stem
            
            content_dict = {
                'id': content_id,
                'title': filename.replace('-', ' ').title(),
                'slug': str(relative_path).replace('\\', '/'),
                'content_type': directory,
                'body': content,
                'source_path': str(md_file)
            }
            
            content_list.append(content_dict)
            logging.info(f"Processed: {md_file}")
            
        except Exception as e:
            logging.error(f"Error reading {md_file}: {e}")
    
    return content_list

async def update_rag_system(content_list: list):
    """
    Updates the RAG system with the provided content.
    """
    logging.info(f"Updating RAG system with {len(content_list)} documents...")
    
    for i, content in enumerate(content_list):
        try:
            # Prepare metadata for the RAG system
            metadata = {
                "title": content['title'],
                "type": content['content_type'],
                "source": f"docs/{content['content_type']}/{content['slug']}.md",
                "path": content['source_path']
            }
            
            # Add content to the enhanced RAG system
            result = await enhanced_rag_service.enhanced_embed_content(
                content['id'], 
                content['body'], 
                metadata
            )
            
            logging.info(f"Processed {i+1}/{len(content_list)}: {content['title']}")
            
        except Exception as e:
            logging.error(f"Error processing content {content['title']}: {e}")
    
    logging.info("RAG system update completed!")

def main():
    logging.info("Starting docs update pipeline...")
    
    # Path to the docs directory
    docs_path = "./docs"
    
    # Read all markdown files from the docs directory
    content_list = read_markdown_files(docs_path)
    
    if not content_list:
        logging.warning("No markdown files found in the docs directory.")
        return
    
    logging.info(f"Found {len(content_list)} markdown files to process.")
    
    # Update the RAG system with the new content
    asyncio.run(update_rag_system(content_list))
    
    logging.info("Docs update pipeline completed successfully!")

if __name__ == "__main__":
    main()