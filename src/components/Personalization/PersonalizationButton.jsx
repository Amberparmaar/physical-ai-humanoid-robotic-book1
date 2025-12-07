import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

const PersonalizationButton = ({ contentId }) => {
  const [isPersonalized, setIsPersonalized] = useState(false);
  const [loading, setLoading] = useState(false);

  const togglePersonalization = async () => {
    setLoading(true);
    
    try {
      const response = await fetch('/api/personalize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: 'current-user-id', // This would come from auth context
          content_id: contentId,
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setIsPersonalized(!isPersonalized);
        // Update the content with personalized version
        document.getElementById(`content-${contentId}`).innerHTML = data.content;
      } else {
        console.error('Failed to toggle personalization');
      }
    } catch (error) {
      console.error('Error toggling personalization:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.personalizationContainer}>
      <button 
        className={`${styles.personalizationButton} ${
          isPersonalized ? styles.active : ''
        }`}
        onClick={togglePersonalization}
        disabled={loading}
        title={isPersonalized ? "Personalization is ON" : "Personalization is OFF"}
      >
        {loading ? 'Loading...' : isPersonalized ? 'Personalized' : 'Personalize Content'}
      </button>
      <div className={styles.personalizationInfo}>
        {isPersonalized 
          ? 'Content has been personalized based on your learning preferences' 
          : 'Enable personalization for content tailored to your learning style'}
      </div>
    </div>
  );
};

export default PersonalizationButton;