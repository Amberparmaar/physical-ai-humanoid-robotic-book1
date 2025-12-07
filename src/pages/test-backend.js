// src/pages/test-backend.js
import React, { useState, useEffect } from 'react';
import Layout from '@theme/Layout';
import { apiService } from '../components/api';

function TestBackend() {
  const [status, setStatus] = useState('Checking connection...');
  const [error, setError] = useState(null);

  useEffect(() => {
    // For this simple test, we'll make a direct fetch to the health endpoint
    // since the apiService doesn't have a health check function yet
    const testConnection = async () => {
      try {
        const response = await fetch('http://localhost:8000/health');
        if (response.ok) {
          const data = await response.json();
          setStatus(`Backend is running! Status: ${data.status}`);
        } else {
          setStatus('Backend is not accessible');
        }
      } catch (err) {
        setError(err.message);
        setStatus('Error connecting to backend');
      }
    };

    testConnection();
  }, []);

  return (
    <Layout title="Backend Test" description="Test backend connection">
      <main>
        <div className="container margin-vert--lg">
          <h1>Backend Connection Test</h1>
          <p>Status: {status}</p>
          {error && <p style={{ color: 'red' }}>Error: {error}</p>}
          <p>Make sure your FastAPI backend is running on port 8000</p>
        </div>
      </main>
    </Layout>
  );
}

export default TestBackend;