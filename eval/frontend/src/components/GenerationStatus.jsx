import React, { useState } from 'react';

function GenerationStatus({ messages }) {
  const [isMinimized, setIsMinimized] = useState(false);
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) {
    // Show just a small button to restore when hidden
    return (
      <button
        onClick={() => setIsVisible(true)}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          padding: '10px 15px',
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          color: '#fff',
          border: 'none',
          borderRadius: '20px',
          cursor: 'pointer',
          fontSize: '14px',
          fontFamily: 'SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
          zIndex: 1000
        }}
      >
        Show Status
      </button>
    );
  }

  return (
    <div 
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        width: isMinimized ? '300px' : '400px',
        height: isMinimized ? '50px' : '300px',
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: '#fff',
        borderRadius: '8px',
        fontFamily: 'SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
        fontSize: '14px',
        zIndex: 1000,
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        transition: 'all 0.3s ease',
        overflow: 'hidden'
      }}
    >
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '10px 15px',
        borderBottom: isMinimized ? 'none' : '1px solid rgba(255, 255, 255, 0.2)',
        fontSize: '16px',
        fontWeight: '500'
      }}>
        <span>Generation Status</span>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setIsMinimized(!isMinimized)}
            style={{
              border: 'none',
              background: 'none',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '16px',
              padding: '4px'
            }}
          >
            {isMinimized ? '▢' : '−'}
          </button>
          <button
            onClick={() => setIsVisible(false)}
            style={{
              border: 'none',
              background: 'none',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '16px',
              padding: '4px'
            }}
          >
            ×
          </button>
        </div>
      </div>

      {/* Content */}
      {!isMinimized && (
        <div style={{
          padding: '15px',
          height: '235px',
          overflowY: 'auto'
        }}>
          {messages.map((msg, index) => (
            <div 
              key={index}
              style={{
                marginBottom: '8px',
                color: msg.includes('Successfully') ? '#4CAF50' : 
                       msg.includes('Error') ? '#f44336' : 
                       msg.includes('Attempt') ? '#2196F3' : 
                       '#fff'
              }}
            >
              {msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default GenerationStatus; 