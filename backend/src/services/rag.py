from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from sqlalchemy.orm import Session
from models import User as db_models
import uuid
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self._client = None
        self._collection_initialized = False

    @property
    def client(self):
        if self._client is None:
            # Initialize Qdrant client only when needed
            if settings.qdrant_path:
                # Use local file-based storage (in-memory or persistent)
                self._client = QdrantClient(
                    path=settings.qdrant_path
                )
            elif settings.qdrant_url and "localhost" not in settings.qdrant_url and "127.0.0.1" not in settings.qdrant_url:
                # Use cloud instance
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                    port=settings.qdrant_port
                )
            else:
                # Use local instance
                self._client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    # Uncomment the following line if using Qdrant cloud
                    # api_key=settings.qdrant_api_key
                )
        return self._client

    @property
    def collection_name(self):
        return settings.qdrant_collection_name

    def _create_collection(self):
        """Create Qdrant collection for textbook content if it doesn't exist"""
        try:
            # Check if collection exists
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
            # Check if the vector size is correct
            vector_config = collection_info.config.params.vectors
            if hasattr(vector_config, 'size') and vector_config.size != 384:
                logger.warning(f"Collection has vector size {vector_config.size}, recreating with 384 dimensions")
                # Delete and recreate collection with correct dimensions
                self.client.delete_collection(self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=384,  # Size for our new embedding approach
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Recreated collection '{self.collection_name}' with 384 dimensions")
        except Exception:
            # Create new collection with 384 dimensions
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=384,  # Size for our new embedding approach
                    distance=qdrant_models.Distance.COSINE
                )
            )
            logger.info(f"Created collection '{self.collection_name}' with 384 dimensions")

    def add_content(self, content_id: int, text: str, metadata: Dict = None):
        """Add content to the vector store"""
        # Ensure collection exists
        if not self._collection_initialized:
            self._create_collection()
            self._collection_initialized = True

        if metadata is None:
            metadata = {}

        # In a real implementation, you would generate embeddings using OpenAI or similar
        # For now, we'll use a placeholder embedding
        embedding = self._generate_embedding(text)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=content_id,
                    vector=embedding,
                    payload={
                        "content_id": content_id,
                        "text": text,
                        "metadata": metadata
                    }
                )
            ]
        )

    def search_content(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for relevant content using vector similarity"""
        # Ensure collection exists
        if not self._collection_initialized:
            self._create_collection()
            self._collection_initialized = True

        query_embedding = self._generate_embedding(query)

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        results = []
        for result in search_results:
            results.append({
                "content_id": result.payload["content_id"],
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload["metadata"]
            })

        return results

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API or fallback method"""
        import openai

        # Use OpenAI API if available
        if settings.openai_api_key:
            try:
                # Try the new OpenAI API format
                from openai import OpenAI
                client = OpenAI(api_key=settings.openai_api_key)
                response = client.embeddings.create(
                    input=text,
                    model="text-embedding-ada-002"
                )
                # Return 384-dimensional embedding to match collection
                embedding_384 = response.data[0].embedding[:384]
                return embedding_384
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"OpenAI embedding failed: {e}. Using fallback method.")

        # Fallback: Use TF-IDF inspired approach that works locally
        import re
        from collections import Counter
        import math

        # Simple TF-IDF inspired approach for local embedding
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return [0.0] * 384  # Return 384-dim vector to match collection

        # Create a vocabulary of common robotics/tech terms
        vocab = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'node', 'ros', 'ros2', 'robot', 'robotics', 'gazebo', 'isaac',
            'nvidia', 'simulation', 'topic', 'service', 'action', 'message',
            'publish', 'subscribe', 'client', 'server', 'communication',
            'humanoid', 'ai', 'physical', 'control', 'perception', 'navigation',
            'vlm', 'vla', 'vision', 'language', 'model', 'system', 'function',
            'code', 'api', 'interface', 'data', 'algorithm', 'framework',
            'learning', 'machine', 'deep', 'neural', 'network', 'training',
            'sensor', 'motor', 'actuator', 'controller', 'feedback',
            'environment', 'action', 'state', 'policy', 'reward', 'agent',
            'environment', 'task', 'behavior', 'motion', 'planning', 'path',
            'trajectory', 'kinematics', 'dynamics', 'inverse', 'forward',
            'coordinate', 'frame', 'transform', 'rotation', 'translation',
            'quaternion', 'euler', 'angle', 'position', 'orientation',
            'velocity', 'acceleration', 'force', 'torque', 'impedance',
            'stiffness', 'compliance', 'impedance', 'admittance',
            'jacobian', 'joints', 'links', 'end', 'effector', 'gripper',
            'manipulator', 'arm', 'leg', 'base', 'mobile', 'wheeled',
            'bipedal', 'quadruped', 'humanoid', 'locomotion', 'walking',
            'running', 'balance', 'stability', 'center', 'mass', 'gravity',
            'gaze', 'attention', 'tracking', 'detection', 'recognition',
            'classification', 'segmentation', 'depth', 'stereo', 'lidar',
            'camera', 'rgb', 'image', 'pixel', 'feature', 'descriptor',
            'matching', 'registration', 'calibration', 'intrinsics', 'extrinsics'
        ]

        # Create TF-IDF style vector
        word_counts = Counter(words)
        total_words = len(words)

        embedding = []
        for word in vocab:
            tf = word_counts.get(word, 0) / total_words  # Term frequency
            # For IDF, we'll use a simplified approach (in a real system, we'd compute this across the whole corpus)
            # Here we'll use a simple heuristic: common words get lower weights
            common_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'}
            idf = 0.1 if word in common_words else 1.0
            tfidf = tf * idf
            embedding.append(tfidf)

        # If vocabulary is smaller than 384, pad with zeros
        while len(embedding) < 384:
            embedding.append(0.0)

        # If larger than 384, truncate
        if len(embedding) > 384:
            embedding = embedding[:384]

        return embedding

    def delete_content(self, content_id: int):
        """Delete content from the vector store"""
        # Ensure collection exists
        if not self._collection_initialized:
            self._create_collection()
            self._collection_initialized = True

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.PointIdsList(
                points=[content_id]
            )
        )

    def update_content(self, content_id: int, text: str, metadata: Dict = None):
        """Update existing content in the vector store"""
        self.delete_content(content_id)
        self.add_content(content_id, text, metadata)


# Global instance
rag_service = RAGService()