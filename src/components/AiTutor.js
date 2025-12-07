// src/components/AiTutor.js
import React, { useState, useRef, useEffect } from 'react';
import { apiService } from './api';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import './AiTutor.css';

const AiTutor = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Add initial welcome message
  useEffect(() => {
    setMessages([
      {
        id: 'welcome',
        text: 'Hello! I\'m your AI tutor for Physical AI & Humanoid Robotics. How can I help you with concepts related to ROS2, Gazebo, NVIDIA Isaac, or VLA models?',
        isUser: false
      }
    ]);
  }, []);

  // Scroll to bottom of messages when they change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (query) => {
    // Add user message to the chat
    const userMessage = {
      id: Date.now().toString(),
      text: query,
      isUser: true
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Query the backend
      const response = await apiService.queryRag(query);

      // Add AI response to the chat
      // According to the backend implementation, response should have 'results' and 'sources'
      let aiResponseText = '';

      if (response.results && response.results.length > 0) {
        // If results exist, format them
        aiResponseText = response.results
          .map(r => r.content || r.content_text || r)
          .filter(text => text) // Remove any undefined/empty results
          .join('\n\n');
      } else if (response.sources && response.sources.length > 0) {
        // If no results but sources exist, show sources
        aiResponseText = `I found relevant information based on these sources: ${response.sources.map(s => s.content_id || s).join(', ')}`;
      } else {
        // Default response if no results or sources
        aiResponseText = 'I found relevant information for your query. Please refer to the appropriate course materials.';
      }

      const aiMessage = {
        id: (Date.now() + 1).toString(),
        text: aiResponseText,
        isUser: false
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      // Add error message to the chat
      console.error('Error in handleSendMessage:', error);
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: `Sorry, I encountered an error processing your request. Please make sure the backend server is running. Error: ${error.message || 'Unknown error'}`,
        isUser: false
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="ai-tutor-container">
      <div className="chat-header">
        <h2>AI Tutor for Physical AI & Humanoid Robotics</h2>
        <p>Ask me anything about ROS2, Gazebo, NVIDIA Isaac, VLA models, and more!</p>
      </div>
      
      <div className="chat-messages">
        {messages.map((message) => (
          <ChatMessage 
            key={message.id} 
            message={message.text} 
            isUser={message.isUser} 
          />
        ))}
        {isLoading && (
          <ChatMessage 
            message="Thinking..." 
            isUser={false} 
          />
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <ChatInput 
        onSendMessage={handleSendMessage} 
        disabled={isLoading} 
      />
    </div>
  );
};

export default AiTutor;