import openai
from typing import List, Dict, Optional
import logging
from config import settings

logger = logging.getLogger(__name__)


# Strict RAG system prompt for all OpenRouter requests
STRICT_RAG_SYSTEM_PROMPT = """You are a strict Retrieval-Augmented Generation (RAG) assistant.

CRITICAL RULES:
- Answer ONLY using the provided book context.
- Do NOT use general knowledge, training data, or assumptions.
- Do NOT infer answers that are not explicitly stated in the context.
- If the answer is not found in the context, respond exactly:
  "The answer is not available in the provided book content."
"""


class OpenRouterService:
    """OpenRouter service for RAG-based chatbot with strict context-only responses"""

    def __init__(self):
        self.client = None
        if settings.openrouter_api_key:
            self.client = openai.AsyncOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url
            )
            logger.info("OpenRouter client initialized")
        else:
            logger.warning("OpenRouter API key not found in environment variables")

    async def generate_response(
        self,
        context: str,
        question: str,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a response using OpenRouter with strict RAG system prompt.

        Args:
            context: Book context (search results from RAG)
            question: User's question
            model: Model to use (defaults to settings.openrouter_model)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Generated response adhering to strict RAG rules
        """
        if not self.client:
            logger.error("OpenRouter client not initialized")
            return "The answer is not available in the provided book content."

        try:
            model = model or settings.openrouter_model

            messages = [
                {"role": "system", "content": STRICT_RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Book Context:\n{context}\n\nUser Question:\n{question}"
                }
            ]

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated response for question: {question[:50]}...")
            return answer

        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            # Hard fallback when API fails
            return "The answer is not available in the provided book content."

    def format_context_from_results(self, search_results: List[Dict], max_results: int = 3) -> str:
        """
        Format search results into a single context string.

        Args:
            search_results: List of search results from RAG
            max_results: Maximum number of results to include

        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant content found in the textbook."

        context_parts = []
        for i, result in enumerate(search_results[:max_results], 1):
            text = result.get('text', '')
            if text:
                context_parts.append(f"[{i}] {text}")

        return "\n\n".join(context_parts)


# Global instance
openrouter_service = OpenRouterService()
