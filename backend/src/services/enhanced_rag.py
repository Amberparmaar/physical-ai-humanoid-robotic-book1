from typing import List, Dict, Optional
from ..services.rag import rag_service
from ..services.context7 import context7_service
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class EnhancedRAGService:
    def __init__(self):
        self.qdrant_rag = rag_service
        self.context7 = context7_service
    
    async def search_content(self, query: str, use_context7: bool = True, limit: int = 5) -> List[Dict]:
        """Enhanced search using both Qdrant and optionally Context7"""
        # First, try the Qdrant-based RAG search
        qdrant_results = self.qdrant_rag.search_content(query, limit)
        
        if use_context7 and settings.context7_api_key:
            try:
                # Also perform a search with Context7
                context7_results = await self.context7.search_content(query)
                
                # Combine and deduplicate results
                combined_results = qdrant_results.copy()
                
                # Add unique results from Context7
                context7_texts = [item.get('text', '') for item in context7_results.get('results', [])]
                for item in context7_results.get('results', []):
                    if item.get('text') not in context7_texts:
                        combined_results.append({
                            'content_id': item.get('id', 'context7-' + str(hash(item.get('text', '')))),
                            'text': item.get('text', ''),
                            'score': item.get('relevance', 0.5),  # Default relevance score
                            'metadata': item.get('metadata', {}),
                            'source': 'context7'
                        })
                
                # Sort by score if available
                combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                return combined_results[:limit]
            except Exception as e:
                logger.warning(f"Context7 search failed, falling back to Qdrant only: {str(e)}")
                return qdrant_results
        else:
            # Only use Qdrant RAG
            return qdrant_results
    
    async def enhanced_embed_content(self, content_id: int, text: str, metadata: Dict = None) -> Dict:
        """Enhanced embedding using both Qdrant and Context7"""
        results = {
            "qdrant_result": None,
            "context7_result": None
        }
        
        try:
            # Add to Qdrant
            self.qdrant_rag.add_content(content_id, text, metadata)
            results["qdrant_result"] = {"status": "success", "content_id": content_id}
        except Exception as e:
            results["qdrant_result"] = {"status": "error", "error": str(e)}
        
        if settings.context7_api_key:
            try:
                # Also add to Context7
                context7_embedding = await self.context7.get_embeddings(text)
                if context7_embedding:
                    results["context7_result"] = {
                        "status": "success", 
                        "embedding_size": len(context7_embedding),
                        "content_id": content_id
                    }
                else:
                    results["context7_result"] = {"status": "error", "error": "Failed to get embeddings"}
            except Exception as e:
                results["context7_result"] = {"status": "error", "error": str(e)}
        
        return results


# Global instance
enhanced_rag_service = EnhancedRAGService()