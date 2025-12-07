from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from .. import schemas
from ..services.context7 import context7_service
import asyncio

router = APIRouter()


@router.post("/search", response_model=Dict[str, Any])
async def context7_search(query: schemas.ChatMessage):
    """Perform search using Context7 MCP"""
    try:
        result = await context7_service.search_content(query.message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context7 search failed: {str(e)}")


@router.post("/embeddings", response_model=Dict[str, Any])
async def get_context7_embeddings(text: schemas.ChatMessage):
    """Get embeddings using Context7 API"""
    try:
        embedding = await context7_service.get_embeddings(text.message)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to get embeddings from Context7")
        
        return {
            "text": text.message,
            "embedding_size": len(embedding),
            "embedding": embedding[:10]  # Return first 10 values as example
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context7 embeddings failed: {str(e)}")


@router.post("/action/{action_name}", response_model=Dict[str, Any])
async def execute_context7_action(action_name: str, params: Dict[str, Any]):
    """Execute a specific action via Context7 MCP"""
    try:
        result = await context7_service.execute_action(action_name, params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context7 action {action_name} failed: {str(e)}")