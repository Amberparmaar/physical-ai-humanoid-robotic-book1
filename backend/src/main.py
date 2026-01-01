import os
import sys

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import engine
from models import Base

from api import auth, chat, context7, query, rag, user

# Initialize FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics API",
    description="Backend API for the Physical AI & Humanoid Robotics textbook platform",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000/physical-ai-humanoid-robotic-book1/"
    ],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
app.include_router(user.router, prefix="/api/v1", tags=["user"])
app.include_router(context7.router, prefix="/api/v1", tags=["context7"])
app.include_router(query.router, prefix="/api", tags=["query"])


@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics API is running!"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Run this function to create tables when needed
def create_database_tables():
    """Call this function to create database tables"""
    Base.metadata.create_all(bind=engine)
