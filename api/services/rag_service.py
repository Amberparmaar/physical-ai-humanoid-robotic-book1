from api.services.embedding_service import EmbeddingService
from api.models.textbook_content import QueryRequest
from typing import List, Dict, Any

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    async def query(self, query: str, top_k: int = 5, user_id: str = None) -> Dict[str, Any]:
        # Get relevant content from vector store
        search_results = await self.embedding_service.query_embeddings(query, top_k)
        
        # If user_id provided, we could personalize results based on learning progress
        if user_id:
            # Here we would fetch user preferences and learning progress to adjust results
            # For now, we'll return the basic search results
            pass
        
        # Format the response
        response = {
            "results": search_results,
            "sources": [result["content_id"] for result in search_results]
        }

        return response