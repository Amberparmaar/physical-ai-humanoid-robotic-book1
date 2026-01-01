// src/components/api.js
// Determine environment - make API calls if backend is accessible
const checkBackendStatus = async () => {
  try {
    const response = await fetch('http://127.0.0.1:8000/health', {
      method: 'GET',
      mode: 'cors',
    });
    return response.ok;
  } catch (error) {
    console.log('Backend not available, using enhanced mock responses');
    return false;
  }
};

// Enhanced mock responses for when backend isn't available
const mockResponses = {
  "hello": "Hello! I'm your AI tutor for Physical AI & Humanoid Robotics. How can I help you with concepts related to ROS2, Gazebo, NVIDIA Isaac, or VLA models?",
  "hi": "Hello! I'm your AI tutor for Physical AI & Humanoid Robotics. How can I help you with concepts related to ROS2, Gazebo, NVIDIA Isaac, or VLA models?",
  "greetings": "Hello! I'm your AI tutor for Physical AI & Humanoid Robotics. How can I help you with concepts related to ROS2, Gazebo, NVIDIA Isaac, or VLA models?",
  "help": "I'm your AI tutor for Physical AI & Humanoid Robotics. I can help you understand concepts related to ROS2, Gazebo, NVIDIA Isaac, VLA models, and other aspects of robotics.",
  "ros2": "ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides services such as hardware abstraction, device drivers, libraries, and more. ROS2 uses a DDS (Data Distribution Service) based communication system, providing better performance and more flexibility than ROS1.",
  "ros": "ROS2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides services such as hardware abstraction, device drivers, libraries, and more. ROS2 uses a DDS (Data Distribution Service) based communication system, providing better performance and more flexibility than ROS1.",
  "gazebo": "Gazebo is a robot simulation environment that allows you to create realistic robot models and test them in a safe virtual space. It provides high-fidelity physics simulation, realistic rendering, and support for various sensors, making it ideal for testing robotic algorithms before deploying to real hardware.",
  "isaac": "The NVIDIA Isaac platform provides tools and libraries for developing GPU-accelerated robotic applications. It includes Isaac ROS for perception and navigation, Isaac Sim for simulation, and Isaac Lab for reinforcement learning for robotics.",
  "nvidia": "The NVIDIA Isaac platform provides tools and libraries for developing GPU-accelerated robotic applications. It includes Isaac ROS for perception and navigation, Isaac Sim for simulation, and Isaac Lab for reinforcement learning for robotics.",
  "vla": "Vision Language Action (VLA) models combine visual perception, language understanding, and action planning for humanoid robots. Examples include RT-2, RT-3, and other transformer-based models that can interpret natural language instructions and execute corresponding robotic actions.",
  "robotics": "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others. It deals with the design, construction, operation, and application of robots, as well as computer systems for their control, sensory feedback, and information processing.",
  "physical ai": "Physical AI refers to artificial intelligence systems that interact directly with the physical world through robotic systems. This includes perception, decision-making, and action execution in real-world environments.",
  "humanoid": "Humanoid robots are robots with human-like form and capabilities, designed for interaction with human-centered environments. Examples include ASIMO, Atlas, and Sophia, which are designed to navigate human spaces and potentially interact with humans.",
  "simulation": "Simulation environments like Gazebo allow for safe testing of robotic algorithms before deploying to real hardware. This reduces risk and allows for extensive testing under various conditions.",
  "control": "Robot control involves algorithms that determine how a robot moves and behaves. This includes trajectory planning, feedback control, and motion control to achieve desired tasks.",
  "navigation": "Robot navigation involves perception, mapping, path planning, and motion control to move a robot from one location to another. Techniques include SLAM (Simultaneous Localization and Mapping).",
  "perception": "Robot perception is the ability of a robot to interpret sensory information about its environment. This includes computer vision, lidar processing, and other sensing modalities.",
  "machine learning": "Machine learning in robotics involves algorithms that allow robots to improve their behavior through experience. This includes supervised learning, reinforcement learning, and unsupervised learning techniques.",
  "reinforcement learning": "Reinforcement learning in robotics involves learning optimal behaviors through trial and error with rewards. This is particularly useful for complex control tasks and manipulation.",
  "manipulation": "Robotic manipulation involves controlling robot arms and end effectors to interact with objects in the environment. This includes grasping, picking, and placing objects.",
  "path planning": "Path planning is the process of determining a route for a robot to move from a starting position to a goal position while avoiding obstacles. Algorithms include A*, Dijkstra, and RRT.",
  "slam": "SLAM (Simultaneous Localization and Mapping) is the computational problem of constructing or updating a map of an unknown environment while simultaneously keeping track of an agent's location within it."
};

export const apiService = {
  // Query the RAG system
  queryRag: async (query, top_k = 5, user_id = null) => {
    // Check if backend is available
    const isBackendAvailable = await checkBackendStatus();

    if (isBackendAvailable) {
      // Backend is available, make API call
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
        console.error('Error querying RAG system, falling back to enhanced mock responses:', error);

        // Fall back to enhanced mock responses
        const query_lower = query.toLowerCase();
        let response_text = "I found relevant information for your query. Please refer to the appropriate course materials.";
        let matched_keyword = null;

        for (const [keyword, response] of Object.entries(mockResponses)) {
          if (query_lower.includes(keyword)) {
            response_text = response;
            matched_keyword = keyword;
            break;
          }
        }

        return {
          results: [
            {
              content_id: matched_keyword ? `mock_content_${matched_keyword}` : "mock_content_general",
              module: "general",
              content: response_text,
              score: 0.9
            }
          ],
          sources: ["enhanced_mock_responses"]
        };
      }
    } else {
      // Backend not available, use enhanced mock responses
      const query_lower = query.toLowerCase();
      let response_text = "I found relevant information for your query. Please refer to the appropriate course materials.";
      let matched_keyword = null;

      // Check for matches in our enhanced responses
      for (const [keyword, response] of Object.entries(mockResponses)) {
        if (query_lower.includes(keyword)) {
          response_text = response;
          matched_keyword = keyword;
          break;
        }
      }

      return {
        results: [
          {
            content_id: matched_keyword ? `mock_content_${matched_keyword}` : "mock_content_general",
            module: "general",
            content: response_text,
            score: 0.9
          }
        ],
        sources: ["enhanced_mock_responses"]
      };
    }
  },

  // Create embeddings for content (if needed)
  createEmbeddings: async (content) => {
    const isBackendAvailable = await checkBackendStatus();

    if (isBackendAvailable) {
      // Backend is available, make API call
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
        // Return mock response when backend call fails
        return { vector_id: "mock_vector_id_123" };
      }
    } else {
      // Backend not available, return mock response
      return { vector_id: "mock_vector_id_123" };
    }
  }
};