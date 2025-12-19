import React from 'react';

function TagDrawer({ tags, onClose, onTagPreference, imageId, position, preferences }) {
  // Helper to normalize tag for comparison (handles whitespace/case differences)
  const normalizeTag = (t) => t?.toLowerCase().trim() || '';

  // Function to check if a tag has a preference
  const getTagPreference = (tag) => {
    const stageTags = preferences?.tags?.[preferences.currentStage] || [];
    // Use normalized comparison for robust matching
    const normalizedTag = normalizeTag(tag);
    const existingPref = stageTags.find(
      t => normalizeTag(t.tag) === normalizedTag && t.source_image === imageId
    );
    return existingPref?.preference || null;
  };

  return (
    <div style={{
      position: 'absolute',
      ...(position.top ? { top: position.bottom + 10 } : { bottom: position.top + 10 }),
      left: position.left,
      width: '300px',
      maxHeight: '300px',
      backgroundColor: 'white',
      boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
      borderRadius: '8px',
      padding: '15px',
      zIndex: 1000,
      overflowY: 'auto',
      fontFamily: 'SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '15px',
        borderBottom: '1px solid #eee',
        paddingBottom: '10px'
      }}>
        <h3 style={{ 
          margin: 0,
          fontSize: '16px',
          fontWeight: '500'
        }}>Image Tags</h3>
        <button
          onClick={onClose}
          style={{
            border: 'none',
            background: 'none',
            fontSize: '20px',
            cursor: 'pointer',
            color: '#666',
            padding: '5px',
            lineHeight: 1
          }}
        >
          ×
        </button>
      </div>
      
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {tags.map((tag, index) => {
          const preference = getTagPreference(tag);
          return (
            <div
              key={index}
              style={{
                padding: '8px 12px',
                backgroundColor: preference === 'positive' ? '#e8f5e9' :
                                preference === 'negative' ? '#ffebee' :
                                '#f5f5f5',
                borderRadius: '6px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'all 0.2s ease',
                fontSize: '14px'
              }}
            >
              <span style={{ flex: 1 }}>{tag}</span>
              <div style={{ 
                display: 'flex', 
                gap: '8px',
                marginLeft: '12px'
              }}>
                <button
                  onClick={() => onTagPreference(tag, 'positive', imageId)}
                  style={{
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    backgroundColor: preference === 'positive' ? '#4caf50' : '#e8f5e9',
                    color: preference === 'positive' ? 'white' : '#4caf50',
                    transition: 'all 0.2s ease'
                  }}
                >
                  👍
                </button>
                <button
                  onClick={() => onTagPreference(tag, 'negative', imageId)}
                  style={{
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    backgroundColor: preference === 'negative' ? '#f44336' : '#ffebee',
                    color: preference === 'negative' ? 'white' : '#f44336',
                    transition: 'all 0.2s ease'
                  }}
                >
                  👎
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default TagDrawer; 