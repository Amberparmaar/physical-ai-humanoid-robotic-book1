import os
import sys

# Add the backend/src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

# Set environment variables to use local storage instead of external services
os.environ.setdefault('QDRANT_PATH', './backend/src/qdrant_data')  # Use local storage
os.environ.setdefault('DATABASE_URL', 'sqlite:///./test.db')  # Use SQLite for testing

# Now import and run the application
from main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)