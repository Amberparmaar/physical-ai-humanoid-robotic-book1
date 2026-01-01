from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


# User schemas
class UserBase(BaseModel):
    email: str
    username: str
    full_name: Optional[str] = None
    is_active: Optional[bool] = True


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    preferences: Optional[str] = None
    learning_path: Optional[str] = None


class User(UserBase):
    id: int
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# UserProfile schemas
class UserProfileBase(BaseModel):
    preferred_language: Optional[str] = "en"
    learning_level: Optional[str] = "beginner"
    specializations: Optional[str] = None
    progress_data: Optional[str] = None


class UserProfileCreate(UserProfileBase):
    user_id: int


class UserProfile(UserProfileBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Content schemas
class ContentBase(BaseModel):
    title: str
    slug: str
    content_type: str
    body: str
    language: Optional[str] = "en"
    is_published: Optional[bool] = False


class ContentCreate(ContentBase):
    pass


class ContentUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    is_published: Optional[bool] = None


class Content(ContentBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# Token schema
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# Chat schemas
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    context_used: Optional[List[str]] = []
    sources: Optional[List[str]] = []