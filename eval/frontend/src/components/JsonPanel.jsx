import React from 'react';

function JsonPanel({ jsonData, onClose, imageId }) {
  // Check if this is tag weights data
  const isTagWeights = jsonData && jsonData.tag_weights !== undefined;
  
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
        }}>
          {isTagWeights ? 'Tag Weights' : 'JSON Script'} - {imageId}
        </h3>
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
      
      {isTagWeights ? (
        // Display tag weights in a nice format
        <div>
          <div style={{
            marginBottom: '15px',
            padding: '10px',
            backgroundColor: '#e3f2fd',
            borderRadius: '6px',
            fontSize: '13px'
          }}>
            <div><strong>Round:</strong> {jsonData.round}</div>
            <div><strong>Concepts with weight &gt; 0:</strong> {jsonData.total_concepts}</div>
          </div>
          
          <div style={{
            backgroundColor: '#f8f9fa',
            border: '1px solid #e9ecef',
            borderRadius: '6px',
            padding: '15px',
            fontSize: '14px',
            fontFamily: 'SF Mono, Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace',
            lineHeight: '1.8',
            whiteSpace: 'pre-wrap',
            overflow: 'auto'
          }}>
            {jsonData.tag_weights}
          </div>
          
          <details style={{ marginTop: '20px' }}>
            <summary style={{
              cursor: 'pointer',
              padding: '10px',
              backgroundColor: '#f5f5f5',
              borderRadius: '4px',
              fontSize: '13px',
              fontWeight: '600'
            }}>
              Show Raw Data
            </summary>
            <div style={{
              marginTop: '10px',
              backgroundColor: '#f8f9fa',
              border: '1px solid #e9ecef',
              borderRadius: '6px',
              padding: '15px',
              fontSize: '12px',
              fontFamily: 'SF Mono, Monaco, "Cascadia Code", "Roboto Mono", Consolas, "Courier New", monospace',
              lineHeight: '1.5',
              whiteSpace: 'pre-wrap',
              overflow: 'auto',
              maxHeight: '300px'
            }}>
              {JSON.stringify(jsonData.raw_data, null, 2)}
            </div>
          </details>
        </div>
      ) : (
        // Display regular JSON
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
      )}
    </div>
  );
}

export default JsonPanel; 