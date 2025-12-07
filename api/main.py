from fastapi import FastAPI
from api.routes import rag

app = FastAPI(title="Physical AI & Humanoid Robotics RAG API")

app.include_router(rag.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Physical AI & Humanoid Robotics RAG API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}