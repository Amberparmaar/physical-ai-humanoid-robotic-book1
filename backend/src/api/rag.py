from fastapi import APIRouter, HTTPException
from typing import List
from .. import schemas
from ..services.rag import rag_service
from ..services.enhanced_rag import enhanced_rag_service
import asyncio

router = APIRouter()


@router.post("/search", response_model=schemas.ChatResponse)
async def search_content(query: schemas.ChatMessage):
    """Search content using the enhanced RAG system with Context7 integration"""
    search_results = await enhanced_rag_service.search_content(query.message)

    if not search_results:
        return schemas.ChatResponse(
            response="No relevant content found in the textbook.",
            context_used=[],
            sources=[]
        )

    # Construct response from search results
    response_text = f"Found {len(search_results)} relevant sections in the textbook:\n\n"
    context_used = []
    sources = []

    for result in search_results:
        response_text += f"- {result['text'][:100]}...\n"
        context_used.append(result['text'][:100] + "...")
        if 'metadata' in result and 'source' in result['metadata']:
            sources.append(result['metadata']['source'])
        elif result.get('source'):
            sources.append(result['source'])

    return schemas.ChatResponse(
        response=response_text,
        context_used=context_used,
        sources=sources
    )


@router.post("/embed", response_model=dict)
async def create_embeddings(content: schemas.Content):
    """Create embeddings for content using both Qdrant and Context7"""
    try:
        # Add content to both Qdrant and Context7 using enhanced service
        metadata = {"title": content.title, "type": content.content_type, "source": f"docs/{content.content_type}/{content.slug}.md"}
        results = await enhanced_rag_service.enhanced_embed_content(content.id, content.body, metadata)

        return {
            "content_id": content.id,
            "qdrant_result": results["qdrant_result"],
            "context7_result": results["context7_result"],
            "vector_size": 1536  # Standard embedding size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")


@router.delete("/embed/{content_id}", response_model=dict)
def delete_embeddings(content_id: int):
    """Remove embeddings for content from vector database"""
    try:
        rag_service.delete_content(content_id)
        return {
            "content_id": content_id,
            "embeddings_deleted": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete embeddings: {str(e)}")


@router.put("/embed/{content_id}", response_model=dict)
async def update_embeddings(content_id: int, content: schemas.Content):
    """Update embeddings for existing content in vector database"""
    try:
        metadata = {"title": content.title, "type": content.content_type, "source": f"docs/{content.content_type}/{content.slug}.md"}
        results = await enhanced_rag_service.enhanced_embed_content(content_id, content.body, metadata)

        return {
            "content_id": content_id,
            "qdrant_result": results["qdrant_result"],
            "context7_result": results["context7_result"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update embeddings: {str(e)}")