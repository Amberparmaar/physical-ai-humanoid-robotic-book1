from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List
from api.models.textbook_content import TextbookContentCreate, TextbookContentResponse, QueryRequest, QueryResponse
from api.services.rag_service import RAGService
from api.services.embedding_service import EmbeddingService
from api.database import database

router = APIRouter()
rag_service = RAGService()
embedding_service = EmbeddingService()

@router.post("/embeddings", summary="Create embeddings for content")
async def create_embeddings(content: TextbookContentCreate):
    try:
        vector_id = await embedding_service.create_embeddings(content)
        return {"vector_id": vector_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", summary="Query the RAG system", response_model=QueryResponse)
async def query_rag(
    query: str = Query(..., description="The query to search"),
    top_k: int = Query(5, description="Number of results to return"),
    user_id: str = Query(None, description="Optional user ID for personalization")
):
    try:
        result = await rag_service.query(query, top_k, user_id)
        # Return both results and sources
        from api.models.textbook_content import TextbookContentResponse
        results = [
            TextbookContentResponse(
                content_id=item["content_id"],
                module=item["module"],
                content=item.get("content", item.get("content_text", "")),
                score=item["score"]
            )
            for item in result["results"]
        ]
        return QueryResponse(results=results, sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))