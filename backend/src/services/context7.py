import aiohttp
import asyncio
from typing import Dict, Any, Optional
from ..config import settings
import logging

logger = logging.getLogger(__name__)

class Context7Service:
    def __init__(self):
        self.api_key = settings.context7_api_key
        self.mcp_host = settings.context7_mcp_host
        self.mcp_port = settings.context7_mcp_port
        self.base_url = f"http://{self.mcp_host}:{self.mcp_port}"
        
        # Headers for API requests
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def search_content(self, query: str) -> Dict[str, Any]:
        """Perform content search using Context7 MCP"""
        try:
            async with aiohttp.ClientSession() as session:
                # This is a placeholder - the actual Context7 API endpoints would be used here
                # For demonstration purposes, we'll show how it would be implemented
                url = f"{self.base_url}/search"  # This would be the actual Context7 search endpoint
                payload = {
                    "query": query,
                    "context": "physical-ai-humanoid-robotics",
                    "limit": 5
                }
                
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"Context7 search failed with status {response.status}")
                        return {"error": f"Search failed with status {response.status}"}
        except Exception as e:
            logger.error(f"Error in Context7 search: {str(e)}")
            return {"error": str(e)}
    
    async def get_embeddings(self, text: str) -> Optional[list]:
        """Get embeddings for text using Context7 API"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/embeddings"  # This would be the actual Context7 embeddings endpoint
                payload = {
                    "input": text,
                    "model": "context7-embeddings"  # Placeholder model name
                }
                
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Return the embedding vector
                        return result.get("data", [])[0].get("embedding", []) if result.get("data") else []
                    else:
                        logger.error(f"Context7 embeddings failed with status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error getting Context7 embeddings: {str(e)}")
            return None
    
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific action using Context7 MCP"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/actions/{action}"  # This would be the actual Context7 actions endpoint
                payload = {
                    "action": action,
                    "parameters": params
                }
                
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        logger.error(f"Context7 action {action} failed with status {response.status}")
                        return {"error": f"Action failed with status {response.status}"}
        except Exception as e:
            logger.error(f"Error executing Context7 action {action}: {str(e)}")
            return {"error": str(e)}


# Global instance
context7_service = Context7Service()