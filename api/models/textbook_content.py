from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

class TextbookContentCreate(BaseModel):
    content_id: str
    module: str
    section: str
    page_number: int
    topic: str
    difficulty_level: str
    content_type: str
    language: str
    content_text: str

class TextbookContentResponse(BaseModel):
    content_id: str
    module: str
    content: str
    score: float

class UserCreate(BaseModel):
    email: str
    name: str
    password: str

class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    role: str
    created_at: datetime

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    results: List[TextbookContentResponse]
    sources: List[dict]