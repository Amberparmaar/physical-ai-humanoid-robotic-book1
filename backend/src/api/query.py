from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.enhanced_rag import enhanced_rag_service

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    user_id: str = None

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    sources: List[str]

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Compatibility endpoint that matches frontend expectations"""
    try:
        # Use the enhanced RAG service to search content
        search_results = await enhanced_rag_service.search_content(request.query, limit=request.top_k)

        if not search_results:
            # Return empty results if nothing found
            return QueryResponse(results=[], sources=[])

        # Format results to match frontend expectations
        formatted_results = []
        sources = set()  # Use a set to avoid duplicates

        for result in search_results:
            # Extract content and metadata
            content = result.get('text', result.get('body', ''))
            metadata = result.get('metadata', {})

            formatted_result = {
                'content_id': result.get('id', ''),
                'module': metadata.get('type', 'unknown'),
                'content': content,
                'score': result.get('score', 0.0),
                'source': metadata.get('source', '')
            }

            formatted_results.append(formatted_result)

            # Add source to set if available
            if metadata.get('source'):
                sources.add(metadata.get('source'))
            elif result.get('source'):
                sources.add(result['source'])

        return QueryResponse(results=formatted_results, sources=list(sources))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")