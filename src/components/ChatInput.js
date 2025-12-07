// src/components/ChatInput.js
import React, { useState } from 'react';
import './ChatInput.css';

const ChatInput = ({ onSendMessage, disabled }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Ask a question about Physical AI & Humanoid Robotics..."
        disabled={disabled}
        className="chat-input"
      />
      <button 
        type="submit" 
        disabled={disabled || !inputValue.trim()}
        className="send-button"
      >
        Send
      </button>
    </form>
  );
};

export default ChatInput;