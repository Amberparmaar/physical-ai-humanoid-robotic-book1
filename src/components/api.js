// src/components/api.js
// Define the backend API URL - adjust this if your backend runs on a different port
const API_BASE_URL = 'http://127.0.0.1:8000/api';

// Service to communicate with the backend API
export const apiService = {
  // Query the RAG system
  queryRag: async (query, top_k = 5, user_id = null) => {
    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          top_k,
          user_id
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error querying RAG system:', error);
      throw error;
    }
  },

  // Create embeddings for content (if needed)
  createEmbeddings: async (content) => {
    try {
      const response = await fetch(`${API_BASE_URL}/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(content),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error creating embeddings:', error);
      throw error;
    }
  }
};