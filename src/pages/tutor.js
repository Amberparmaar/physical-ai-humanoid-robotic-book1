// src/pages/tutor.js
import React from 'react';
import Layout from '@theme/Layout';
import './tutor.module.css';

function Tutor() {
  return (
    <Layout title="AI Tutor" description="Physical AI & Humanoid Robotics AI Tutor">
      <main className="tutor-page">
        <div className="container margin-vert--lg text--center">
          <h1>AI Tutor</h1>
          <p>The AI Tutor is now available as a floating chatbot!</p>
          <p>Look for the chat icon in the bottom right corner of the screen to interact with the AI tutor.</p>
        </div>
      </main>
    </Layout>
  );
}

export default Tutor;