import React from 'react';

function NavigationBar({ parallelStages, onStageClick, currentParallelStage }) {
  const stages = ['impression', 'spatial', 'ambient'];
  
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '30px',
      padding: '20px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px',
      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
    }}>
      <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
        {stages.map((stage, index) => {
          const stageData = parallelStages[stage];
          const isCompleted = stageData?.completed || false;
          const hasSelection = stageData?.selectedImage;
          const isCurrent = currentParallelStage === stage;
          const hasImages = stageData?.images && stageData.images.length > 0;
          
          return (
            <div key={stage} style={{ display: 'flex', alignItems: 'center' }}>
              <button
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  console.log(`NavigationBar: ${stage} clicked`);
                  
                  if (onStageClick) {
                    onStageClick(stage);
                  } else {
                    console.error('NavigationBar: onStageClick function not provided!');
                  }
                }}
                style={{
                  padding: '15px 25px',
                  borderRadius: '8px',
                  border: isCurrent ? '3px solid #007bff' : '2px solid #ddd',
                  backgroundColor: isCurrent 
                    ? '#e7f3ff' 
                    : (isCompleted ? '#d4edda' : '#f8f9fa'),
                  color: isCurrent 
                    ? '#007bff' 
                    : (isCompleted ? '#155724' : '#666'),
                  cursor: 'pointer',
                  fontWeight: isCurrent ? 'bold' : 'normal',
                  fontSize: '16px',
                  transition: 'all 0.3s ease',
                  position: 'relative',
                  minWidth: '120px',
                  textAlign: 'center',
                  // Ensure button is always clickable
                  pointerEvents: 'auto',
                  zIndex: 10
                }}
                type="button"
                disabled={false} // Always enabled for testing
              >
                {stage.charAt(0).toUpperCase() + stage.slice(1)}
                
                {/* Status indicators */}
                {isCompleted && (
                  <span style={{
                    position: 'absolute',
                    top: '-8px',
                    right: '-8px',
                    backgroundColor: '#28a745',
                    color: 'white',
                    borderRadius: '50%',
                    width: '24px',
                    height: '24px',
                    fontSize: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold'
                  }}>
                    ✓
                  </span>
                )}
                
                {hasSelection && (
                  <span style={{
                    position: 'absolute',
                    bottom: '-8px',
                    right: '-8px',
                    backgroundColor: '#ffc107',
                    color: '#000',
                    borderRadius: '50%',
                    width: '20px',
                    height: '20px',
                    fontSize: '10px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold'
                  }}>
                    ★
                  </span>
                )}
                
                {/* Show image count if available */}
                {hasImages && (
                  <div style={{
                    position: 'absolute',
                    bottom: '-20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    fontSize: '11px',
                    color: '#666',
                    whiteSpace: 'nowrap'
                  }}>
                    {stageData.images.length} images
                  </div>
                )}
              </button>
              
              {/* Connector line */}
              {index < stages.length - 1 && (
                <div style={{
                  width: '40px',
                  height: '3px',
                  backgroundColor: isCompleted ? '#28a745' : '#ddd',
                  margin: '0 10px',
                  borderRadius: '2px'
                }} />
              )}
            </div>
          );
        })}
        
        {/* Final stage button */}
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{
            width: '40px',
            height: '3px',
            backgroundColor: '#ddd',
            margin: '0 10px',
            borderRadius: '2px'
          }} />
          <div style={{
            padding: '15px 25px',
            borderRadius: '8px',
            border: '2px solid #17a2b8',
            backgroundColor: '#e1f7fa',
            color: '#0c5460',
            fontWeight: 'bold',
            fontSize: '16px',
            minWidth: '120px',
            textAlign: 'center'
          }}>
            Final
          </div>
        </div>
      </div>
    </div>
  );
}

export default NavigationBar; 