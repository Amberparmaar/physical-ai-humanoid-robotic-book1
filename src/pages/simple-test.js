// src/pages/simple-test.js
import React, { useState } from 'react';
import Layout from '@theme/Layout';

function SimpleTest() {
  const [status, setStatus] = useState('');
  const [backendResponse, setBackendResponse] = useState('');

  const testBackend = async () => {
    setStatus('Testing connection...');
    try {
      // Test health endpoint
      const response = await fetch('http://127.0.0.1:8000/health');
      if (response.ok) {
        const data = await response.json();
        setBackendResponse(`Health check: ${JSON.stringify(data)}`);
        
        // Now try the query endpoint with a simple request
        const queryResponse = await fetch('http://127.0.0.1:8000/api/query', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: 'Hello',
            top_k: 1
          })
        });
        
        if (queryResponse.ok) {
          const queryData = await queryResponse.json();
          setBackendResponse(prev => prev + `\nQuery response: ${JSON.stringify(queryData)}`);
          setStatus('Success! Backend is accessible.');
        } else {
          setStatus(`Query endpoint failed with status: ${queryResponse.status}`);
        }
      } else {
        setStatus(`Health endpoint failed with status: ${response.status}`);
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
    }
  };

  return (
    <Layout title="Simple Backend Test">
      <main>
        <div className="container margin-vert--lg">
          <h1>Simple Backend Connection Test</h1>
          <button onClick={testBackend} style={{ padding: '10px 20px', marginBottom: '20px' }}>
            Test Backend Connection
          </button>
          <p><strong>Status:</strong> {status}</p>
          <p><strong>Response:</strong> {backendResponse}</p>
          <p>Make sure your backend is running with: <code>uvicorn api.main:app --reload --port 8000</code></p>
          <p>And your Docker containers are running: <code>docker-compose up -d</code></p>
        </div>
      </main>
    </Layout>
  );
}

export default SimpleTest;