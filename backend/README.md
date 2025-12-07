# Physical AI & Humanoid Robotics - Backend

This is the backend component of the Physical AI & Humanoid Robotics textbook platform, built with FastAPI. It provides the API layer, database integration, RAG (Retrieval Augmented Generation) functionality, and Context7 integration.

## Features

- **User Authentication**: Registration, login, and profile management
- **RAG System**: Integration with Qdrant for vector search and retrieval
- **Context7 Integration**: MCP and API tools for enhanced functionality
- **Content Management**: CRUD operations for textbook content
- **Chat Interface**: AI-powered chatbot with textbook knowledge

## Architecture

- **API Layer**: FastAPI with Pydantic schemas
- **Database**: PostgreSQL with SQLAlchemy ORM (optimized for Neon)
- **Vector Store**: Qdrant for semantic search
- **Authentication**: JWT-based with secure password hashing
- **AI Integration**: OpenAI for content generation (optional)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables in `.env`:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/physical_ai_humanoid_robotics
   SECRET_KEY=your-secret-key-change-in-production
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION_NAME=textbook_content
   OPENAI_API_KEY=your-openai-api-key  # Optional
   CONTEXT7_API_KEY=your-context7-api-key  # Optional
   CONTEXT7_MCP_HOST=localhost
   CONTEXT7_MCP_PORT=8000
   ```

3. Run database migrations:
   ```bash
   # If using alembic
   cd src
   alembic upgrade head
   ```

4. Start the server:
   ```bash
   uvicorn backend.src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Docker Compose Setup

Alternatively, use the provided docker-compose configuration to run the entire stack:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database
- Qdrant vector store
- Docusaurus documentation site
- Backend API server

## API Endpoints

- `GET /` - Health check
- `POST /api/v1/register` - User registration
- `POST /api/v1/login` - User login
- `GET /api/v1/profile` - Get user profile
- `POST /api/v1/chat` - Chat with the textbook AI
- `POST /api/v1/search` - Search textbook content
- `POST /api/v1/embed` - Create embeddings for content

## Context7 Integration

The system includes hooks for Context7 MCP and API integration:
- Context7 search capabilities
- Enhanced embeddings
- Action execution through Context7

## Technologies Used

- FastAPI - Web framework
- SQLAlchemy - ORM
- Qdrant - Vector database
- PostgreSQL - Relational database (with Neon compatibility)
- JWT - Authentication
- OpenAI API - AI responses (optional)
- Context7 - MCP and API tools