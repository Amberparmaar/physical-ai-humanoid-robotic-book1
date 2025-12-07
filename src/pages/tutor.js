// src/pages/tutor.js
import React from 'react';
import Layout from '@theme/Layout';
import AiTutor from '../components/AiTutor';
import '../components/AiTutor.css';

function Tutor() {
  return (
    <Layout title="AI Tutor" description="Physical AI & Humanoid Robotics AI Tutor">
      <main className="tutor-page">
        <div className="container margin-vert--lg">
          <AiTutor />
        </div>
      </main>
    </Layout>
  );
}

export default Tutor;