from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from api.models.textbook_content import TextbookContentCreate
import os
import uuid
from typing import List, Dict, Any

class EmbeddingService:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ensure collection exists
        self._create_collection()
    
    def _create_collection(self):
        try:
            self.client.get_collection("textbook_content")
        except:
            self.client.create_collection(
                collection_name="textbook_content",
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
            )
    
    async def create_embeddings(self, content: TextbookContentCreate) -> str:
        # Generate embedding using OpenAI
        response = self.openai_client.embeddings.create(
            input=content.content_text,
            model="text-embedding-3-large"
        )
        embedding = response.data[0].embedding
        
        # Prepare metadata
        payload = {
            "content_id": content.content_id,
            "module": content.module,
            "section": content.section,
            "page_number": content.page_number,
            "topic": content.topic,
            "difficulty_level": content.difficulty_level,
            "content_type": content.content_type,
            "language": content.language,
            "created_at": content.page_number  # Using page number as simple timestamp
        }
        
        # Generate a unique ID for the vector
        vector_id = str(uuid.uuid4())
        
        # Upsert the vector into Qdrant
        self.client.upsert(
            collection_name="textbook_content",
            points=[models.PointStruct(
                id=vector_id,
                vector=embedding,
                payload=payload
            )]
        )
        
        return vector_id
    
    async def query_embeddings(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Generate embedding for the query
        response = self.openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        
        # Search in Qdrant
        search_result = self.client.search(
            collection_name="textbook_content",
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        # Format results
        results = []
        for hit in search_result:
            result = {
                "content_id": hit.payload["content_id"],
                "module": hit.payload["module"],
                "content": hit.payload["content_text"] if "content_text" in hit.payload else hit.payload.get("content", ""),
                "score": hit.score,
                "section": hit.payload["section"],
                "topic": hit.payload["topic"],
                "difficulty_level": hit.payload["difficulty_level"],
                "language": hit.payload["language"]
            }
            results.append(result)

        return results