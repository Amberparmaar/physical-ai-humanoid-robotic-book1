@echo off
echo Setting up environment and starting backend server...

REM Navigate to the backend directory
cd /d "C:\Users\hp\Desktop\physical-ai-humanoid-robotics\backend"

REM Create a virtual environment (if not already created)
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Install requirements
echo Installing required packages...
pip install fastapi uvicorn sqlalchemy psycopg2-binary python-jose passlib python-multipart alembic pydantic qdrant-client openai aiohttp async-lru

REM Navigate to the src directory and start the server
cd src

echo Starting backend server on port 8000...
python -c "import sys; sys.path.insert(0, '.'); from main import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000)"