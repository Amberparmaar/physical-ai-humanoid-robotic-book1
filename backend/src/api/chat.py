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
import openai
from config import settings
import asyncio
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()

# Set OpenAI API key if available
if settings.openai_api_key:
    openai.api_key = settings.openai_api_key


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
    """Chat with the textbook AI assistant using enhanced RAG with Context7"""
    # Query the enhanced RAG system for relevant content
    search_results = await enhanced_rag_service.search_content(message.message)

    if not search_results and not settings.openai_api_key:
        # If no results and no OpenAI, return a simple response
        response = f"Thanks for your message: '{message.message}'. I couldn't find specific content in the textbook, but feel free to ask about ROS2, Gazebo, NVIDIA Isaac, or VLA models."
        return schemas.ChatResponse(
            response=response,
            context_used=[],
            sources=[]
        )

    if settings.openai_api_key:
        # Use OpenAI to generate a contextual response based on RAG results
        try:
            # Format context from search results
            context_text = "\n".join([result['text'] for result in search_results[:3]])  # Use top 3 results

            prompt = f"""
            You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
            Answer the user's question based on the following context from the textbook.
            If the context doesn't contain relevant information, politely say you don't know
            and suggest checking the appropriate section of the textbook.

            Context: {context_text}

            User question: {message.message}

            Answer:
            """

            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )

            response = completion.choices[0].message['content'].strip()
        except Exception as e:
            # Fallback if OpenAI fails
            response = f"I encountered an issue processing your request: {str(e)}. Thanks for your message: '{message.message}'."
    else:
        # No OpenAI available, construct simple response from RAG results
        if search_results:
            response = f"Based on the textbook: {search_results[0]['text'][:200]}..."
        else:
            response = f"Thanks for your message: '{message.message}'. I couldn't find specific content in the textbook, but feel free to ask about ROS2, Gazebo, NVIDIA Isaac, or VLA models."

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