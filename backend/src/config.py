import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application settings
    app_name: str = "Physical AI & Humanoid Robotics API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database settings
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/physical_ai_humanoid_robotics"
    )

    # JWT settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Qdrant settings
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", 6333))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

    # OpenAI settings
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Context7 settings
    context7_api_key: Optional[str] = os.getenv("CONTEXT7_API_KEY")
    context7_mcp_host: str = os.getenv("CONTEXT7_MCP_HOST", "localhost")
    context7_mcp_port: int = int(os.getenv("CONTEXT7_MCP_PORT", 8000))

    model_config = {"env_file": ".env"}


settings = Settings()