from fastapi import APIRouter, Depends
from typing import List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import ChatMessage, ChatResponse
from services import auth
from models import User
from services.rag import rag_service
from services.enhanced_rag import enhanced_rag_service
from services.openrouter import openrouter_service
from config import settings
import asyncio
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()

security = HTTPBearer()

async def get_current_user_from_token(token: HTTPAuthorizationCredentials = Depends(security)):
    from database import SessionLocal
    db = SessionLocal()
    try:
        user = auth.get_current_user(token.credentials)
        if not user:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    finally:
        db.close()

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(
    message: ChatMessage,
    current_user: User = Depends(get_current_user_from_token)
):
    """Chat with the textbook AI assistant using enhanced RAG with OpenRouter"""
    # Query the enhanced RAG system for relevant content
    search_results = await enhanced_rag_service.search_content(message.message)

    # Format context from search results
    context_text = openrouter_service.format_context_from_results(search_results, max_results=3)

    if settings.openrouter_api_key:
        # Use OpenRouter with strict RAG system prompt to generate response
        response = await openrouter_service.generate_response(
            context=context_text,
            question=message.message,
            max_tokens=500,
            temperature=0.7
        )
    else:
        # Hard fallback when OpenRouter is not available
        if search_results:
            response = f"Based on the textbook: {search_results[0]['text'][:300]}..."
        else:
            response = "The answer is not available in the provided book content."

    # Extract context used and sources from search results
    context_used = [result['text'][:100] + "..." for result in search_results[:3]]
    sources = []
    for result in search_results:
        if 'metadata' in result and 'source' in result['metadata']:
            sources.append(result['metadata']['source'])
        elif result.get('source'):
            sources.append(result['source'])

    return schemas.ChatResponse(
        response=response,
        context_used=context_used,
        sources=list(set(sources))  # Remove duplicates
    )


@router.get("/history", response_model=List[ChatMessage])
def get_chat_history(
    current_user: User = Depends(get_current_user_from_token)
):
    """Get the user's chat history"""
    # This is a placeholder - in a real implementation, this would
    # fetch chat history from the database
    return [
        ChatMessage(message="Hello, what is ROS2?", user_id=current_user.id),
        ChatMessage(message="How does Gazebo work?", user_id=current_user.id)
    ]