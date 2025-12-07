from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from typing import List, Dict, Optional
from ..config import settings
from sqlalchemy.orm import Session
from .. import models as db_models
import uuid
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self._client = None
        self._collection_initialized = False

    @property
    def client(self):
        if self._client is None:
            # Initialize Qdrant client only when needed
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                # Uncomment the following line if using Qdrant cloud
                # api_key=settings.qdrant_api_key
            )
        return self._client

    @property
    def collection_name(self):
        return settings.qdrant_collection_name

    def _create_collection(self):
        """Create Qdrant collection for textbook content if it doesn't exist"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except Exception:
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=1536,  # Standard OpenAI embedding size
                    distance=qdrant_models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection_name}'")

    def add_content(self, content_id: int, text: str, metadata: Dict = None):
        """Add content to the vector store"""
        # Ensure collection exists
        if not self._collection_initialized:
            self._create_collection()
            self._collection_initialized = True

        if metadata is None:
            metadata = {}

        # In a real implementation, you would generate embeddings using OpenAI or similar
        # For now, we'll use a placeholder embedding
        embedding = self._generate_embedding(text)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=content_id,
                    vector=embedding,
                    payload={
                        "content_id": content_id,
                        "text": text,
                        "metadata": metadata
                    }
                )
            ]
        )

    def search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant content using vector similarity"""
        # Ensure collection exists
        if not self._collection_initialized:
            self._create_collection()
            self._collection_initialized = True

        query_embedding = self._generate_embedding(query)

        search_results = self.client.search(
            collection_name=self.collection_name,
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

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder implementation)"""
        # In a real implementation, you would use OpenAI's embedding API
        # or another embedding model to create the actual vector
        # For now, we return a placeholder vector
        import random
        return [random.random() for _ in range(1536)]

    def delete_content(self, content_id: int):
        """Delete content from the vector store"""
        # Ensure collection exists
        if not self._collection_initialized:
            self._create_collection()
            self._collection_initialized = True

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=[content_id]
            )
        )

    def update_content(self, content_id: int, text: str, metadata: Dict = None):
        """Update existing content in the vector store"""
        self.delete_content(content_id)
        self.add_content(content_id, text, metadata)


# Global instance
rag_service = RAGService()