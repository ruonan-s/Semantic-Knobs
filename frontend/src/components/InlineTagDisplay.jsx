import React from 'react';

function InlineTagDisplay({ tags, imageId, onTagPreference, preferences }) {
  console.log('[InlineTagDisplay] RENDER:', {
    imageId,
    currentStage: preferences?.currentStage,
    tagsCount: preferences?.tags?.[preferences?.currentStage]?.length || 0
  });
  
  // Helper to get tag preference
  const getTagPreference = (tag) => {
    const currentStage = preferences?.currentStage;
    const stageTags = preferences?.tags?.[currentStage] || [];
    
    const found = stageTags.find(item => item.tag === tag && item.source_image === imageId);
    console.log(`[InlineTagDisplay] getTagPreference("${tag}"):`, found?.preference || null);
    return found?.preference || null;
  };

  if (!tags || tags.length === 0) {
    return (
      <div style={{
        marginTop: '10px',
        padding: '8px',
        backgroundColor: '#f8f9fa',
        borderRadius: '4px',
        fontSize: '12px',
        color: '#6c757d',
        textAlign: 'center'
      }}>
        No tags available
      </div>
    );
  }

  return (
    <div style={{
      marginTop: '10px',
      padding: '8px',
      backgroundColor: '#f8f9fa',
      borderRadius: '4px',
      maxHeight: '200px',
      overflowY: 'auto'
    }}>
      <div style={{
        fontSize: '12px',
        fontWeight: '500',
        marginBottom: '6px',
        color: '#495057'
      }}>
        Visual Tags ({tags.length})
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {tags.map((tag, index) => {
          const preference = getTagPreference(tag);
          return (
            <div
              key={index}
              style={{
                padding: '4px 8px',
                backgroundColor: preference === 'positive' ? '#e8f5e9' :
                                preference === 'negative' ? '#ffebee' :
                                '#ffffff',
                borderRadius: '4px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                transition: 'all 0.2s ease',
                fontSize: '11px',
                border: '1px solid #dee2e6'
              }}
            >
              <span style={{ flex: 1, marginRight: '8px' }}>{tag}</span>
              <div style={{ 
                display: 'flex', 
                gap: '4px',
                flexShrink: 0
              }}>
                <button
                  onClick={() => onTagPreference(tag, 'positive', imageId)}
                  style={{
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    padding: '2px 4px',
                    borderRadius: '3px',
                    backgroundColor: preference === 'positive' ? '#4caf50' : '#e8f5e9',
                    color: preference === 'positive' ? 'white' : '#4caf50',
                    transition: 'all 0.2s ease',
                    fontSize: '10px'
                  }}
                  title="Like this tag"
                >
                  👍
                </button>
                <button
                  onClick={() => onTagPreference(tag, 'negative', imageId)}
                  style={{
                    border: 'none',
                    background: 'none',
                    cursor: 'pointer',
                    padding: '2px 4px',
                    borderRadius: '3px',
                    backgroundColor: preference === 'negative' ? '#f44336' : '#ffebee',
                    color: preference === 'negative' ? 'white' : '#f44336',
                    transition: 'all 0.2s ease',
                    fontSize: '10px'
                  }}
                  title="Dislike this tag"
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

export default InlineTagDisplay;


