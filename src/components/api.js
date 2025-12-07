// src/components/api.js
// Determine environment - only make API calls when explicitly in development mode
const isDevelopment = typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

// Mock responses for production deployment
const mockResponses = {
  "hello": "Hello! I'm your AI tutor for Physical AI & Humanoid Robotics. How can I help you?",
  "ros2": "ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides services such as hardware abstraction, device drivers, libraries, and more.",
  "gazebo": "Gazebo is a robot simulation environment that allows you to create realistic robot models and test them in a safe virtual space.",
  "isaac": "The NVIDIA Isaac platform provides tools and libraries for developing GPU-accelerated robotic applications.",
  "vla": "Vision Language Action (VLA) models combine visual perception, language understanding, and action planning for humanoid robots.",
  "robotics": "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others.",
  "physical ai": "Physical AI refers to artificial intelligence systems that interact directly with the physical world through robotic systems.",
  "humanoid": "Humanoid robots are robots with human-like form and capabilities, designed for interaction with human-centered environments.",
  "simulation": "Simulation environments like Gazebo allow for safe testing of robotic algorithms before deploying to real hardware."
};

export const apiService = {
  // Query the RAG system
  queryRag: async (query, top_k = 5, user_id = null) => {
    if (isDevelopment) {
      // Only make API calls in development mode
      try {
        const response = await fetch('http://127.0.0.1:8000/api/query', {
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
    } else {
      // Always return mock responses when deployed to avoid any security issues
      const query_lower = query.toLowerCase();
      let response_text = "I found relevant information for your query. Please refer to the appropriate course materials.";

      for (const [keyword, response] of Object.entries(mockResponses)) {
        if (query_lower.includes(keyword)) {
          response_text = response;
          break;
        }
      }

      return {
        results: [
          {
            content_id: "mock_content_1",
            module: "mock_module",
            content: response_text,
            score: 0.9
          }
        ],
        sources: ["mock_source_1"]
      };
    }
  },

  // Create embeddings for content (if needed)
  createEmbeddings: async (content) => {
    if (isDevelopment) {
      // Only make API calls in development mode
      try {
        const response = await fetch('http://127.0.0.1:8000/api/embeddings', {
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
    } else {
      // Always return mock response when deployed to avoid any security issues
      return { vector_id: "mock_vector_id_123" };
    }
  }
};