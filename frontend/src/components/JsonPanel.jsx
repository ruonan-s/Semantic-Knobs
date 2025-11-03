import React from 'react';

function JsonPanel({ jsonData, onClose, imageId }) {
  return (
    <div style={{
      position: 'fixed',
      right: 0,
      top: 0,
      bottom: 0,
      width: '500px',
      backgroundColor: 'white',
      boxShadow: '-2px 0 5px rgba(0,0,0,0.1)',
      padding: '20px',
      zIndex: 1000,
      overflowY: 'auto',
      fontFamily: 'SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        borderBottom: '1px solid #eee',
        paddingBottom: '15px'
      }}>
        <h3 style={{ 
          margin: 0,
          fontSize: '18px',
          fontWeight: '600'
        }}>JSON Script - {imageId}</h3>
        <button
          onClick={onClose}
          style={{
            border: 'none',
            background: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            color: '#666',
            padding: '5px',
            lineHeight: 1
          }}
        >
          ×
        </button>
      </div>
      
      <div style={{
        backgroundColor: '#f8f9fa',
        border: '1px solid #e9ecef',
        borderRadius: '6px',
        padding: '15px',
        fontSize: '14px',
        fontFamily: 'SF Mono, Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace',
        lineHeight: '1.5',
        whiteSpace: 'pre-wrap',
        overflow: 'auto'
      }}>
        {JSON.stringify(jsonData, null, 2)}
      </div>
    </div>
  );
}

export default JsonPanel; 