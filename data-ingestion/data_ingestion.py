import os
import logging
import requests
import cohere
import time
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
from qdrant_client import QdrantClient, models
from trafilatura import extract

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SITEMAP_URL = "../frontend/build/sitemap.xml"
COLLECTION_NAME = "ai_robotics_book"
COHERE_MODEL = "embed-english-v3.0"
VECTOR_SIZE = 1024

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)

def fetch_sitemap_urls(file_path):
    """
    Reads and parses a local sitemap file to extract all URLs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        root = ET.fromstring(content)
        urls = [elem.text for elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')]
        logging.info(f"Found {len(urls)} URLs in the sitemap.")
        if not urls:
            logging.warning("No URLs found in the sitemap.")
            return []
        return urls
    except FileNotFoundError:
        logging.error(f"Sitemap file not found at {file_path}")
        return []
    except ET.ParseError as e:
        logging.error(f"Error parsing sitemap XML: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return []

def _url_to_local_path(sitemap_url, base_dir="../frontend/build"):
    """
    Converts a sitemap URL to a local file path within the Docusaurus build directory.
    """
    # Remove the base URL part from the sitemap URL
    path = sitemap_url.replace("https://areebazafarchohan.github.io", "").lstrip('/')

    # Handle different path structures
    if not path: # Root URL
        local_path = os.path.join(base_dir, "index.html")
    elif path.endswith('/'): # Directory path (e.g., /about/)
        local_path = os.path.join(base_dir, path, "index.html")
    elif '.' in os.path.basename(path): # File path (e.g., /some-file.html)
        local_path = os.path.join(base_dir, path)
    else: # Directory path without trailing slash (e.g., /about)
        local_path = os.path.join(base_dir, path, "index.html")

    return local_path

def extract_text_from_url(url):
    """
    Extracts clean text from a local HTML file.
    """
    local_file_path = _url_to_local_path(url)
    try:
        with open(local_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        text = extract(content)
        if not text:
            logging.warning(f"No text extracted from {local_file_path}")
            return None
        return text
    except FileNotFoundError:
        logging.error(f"Local file not found for URL {url} at path {local_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error extracting text from {local_file_path} (URL: {url}): {e}")
        return None

def chunk_text(text, chunk_size=1200):
    """
    Splits text into chunks of a specified size, prioritizing sentence boundaries.
    """
    chunks = []
    while len(text) > chunk_size:
        last_period = text[:chunk_size].rfind('. ')
        if last_period == -1:
            last_period = chunk_size
        chunks.append(text[:last_period+1])
        text = text[last_period+1:]
    chunks.append(text)
    return chunks

def get_embedding(text, model):
    """
    Generates an embedding for a text chunk with retry logic and rate limiting.
    """
    for i in range(3):
        try:
            time.sleep(0.6) # Rate limit: 100 calls per minute (60/100 = 0.6 seconds per call)
            response = co.embed(texts=[text], model=model, input_type="search_document")
            return response.embeddings[0]
        except cohere.errors.TooManyRequestsError as e:
            logging.warning(f"Cohere API rate limit hit: {e}. Retrying in {2 ** i} seconds. Attempt {i+1} of 3.")
            time.sleep(2 ** i)
        except cohere.core.api_error.ApiError as e: # Catch other Cohere API errors
            logging.warning(f"Cohere API error: {e}. Retrying in {2 ** i} seconds. Attempt {i+1} of 3.")
            time.sleep(2 ** i)
    logging.error(f"Failed to get embedding for chunk after 3 attempts.")
    return None

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def setup_collection():
    """
    Checks if the collection exists and prompts the user to recreate it if it does.
    """
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
        recreate = input("Do you want to delete and recreate the collection? (y/n): ")
        if recreate.lower() != 'y':
            logging.info("Exiting without any changes.")
            exit()
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        logging.info(f"Recreated collection '{COLLECTION_NAME}'.")

    except Exception as e:
        if "not found" in str(e).lower():
            logging.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating it.")
            qdrant_client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
            )
            logging.info(f"Created collection '{COLLECTION_NAME}'.")

def process_and_upsert_data(urls):
    """
    Processes URLs, chunks text, generates embeddings, and upserts to Qdrant.
    """
    total_chunks_processed = 0
    points = []
    point_id_counter = 0

    for i, url in enumerate(urls):
        logging.info(f"Processing URL {i+1}/{len(urls)}: {url}")
        text = extract_text_from_url(url)
        if not text:
            continue

        chunks = chunk_text(text)
        logging.info(f"Extracted {len(chunks)} chunks from the URL.")

        for chunk_id, chunk in enumerate(chunks):
            embedding = get_embedding(chunk, COHERE_MODEL)
            if embedding is None:
                continue

            points.append(
                models.PointStruct(
                    id=point_id_counter,
                    vector=embedding,
                    payload={"url": url, "text": chunk, "chunk_id": chunk_id},
                )
            )
            point_id_counter += 1

            if len(points) >= 50:  # Batch upsert
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    wait=True,
                    points=points,
                )
                total_chunks_processed += len(points)
                logging.info(f"Upserted {len(points)} points to Qdrant collection '{COLLECTION_NAME}'. Total: {total_chunks_processed}")
                points = []

    # Upsert any remaining points
    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points,
        )
        total_chunks_processed += len(points)
        logging.info(f"Upserted {len(points)} remaining points to Qdrant collection '{COLLECTION_NAME}'. Total: {total_chunks_processed}")

    return total_chunks_processed

def main():
    logging.info("Starting Qdrant Ingestion Pipeline...")

    # Phase 1: Setup collection
    setup_collection()

    # Phase 2: Fetch sitemap URLs
    urls = fetch_sitemap_urls(SITEMAP_URL)
    if not urls:
        logging.error("No URLs to process. Exiting.")
        return

    # Phase 3: Process data and upsert to Qdrant
    total_chunks = process_and_upsert_data(urls)
    logging.info(f"Successfully ingested {total_chunks} chunks into the Qdrant collection.")

    # Phase 4: Validate data in Qdrant
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Validation complete: Found {collection_info.points_count} points in the collection.")
    except Exception as e:
        logging.error(f"Error during validation: {e}")

if __name__ == "__main__":
    main()