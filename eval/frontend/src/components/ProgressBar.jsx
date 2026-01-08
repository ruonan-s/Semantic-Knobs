import React from 'react';

function ProgressBar({ currentStage }) {
  // Don't show progress bar for landing stage
  if (currentStage === 'landing') {
    return null;
  }

  // Simplified stages for eval: Exploration → Slider Generation (no refinement)
  const stages = [
    { id: 'impression', label: 'Exploration' },
    { id: 'slider_generation', label: 'Rank' }
  ];
  
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '30px',
      padding: '20px',
      backgroundColor: '#f8f9fa',
      borderRadius: '8px'
    }}>
      <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
        {stages.map((stage, index) => {
          const isActive = currentStage === stage.id;
          const stageIdx = stages.findIndex(s => s.id === currentStage);
          const isCompleted = stageIdx > index;
          
          return (
            <div key={stage.id} style={{ display: 'flex', alignItems: 'center' }}>
              <div
                style={{
                  padding: '12px 20px',
                  borderRadius: '8px',
                  border: isActive ? '2px solid #007bff' : '1px solid #ddd',
                  backgroundColor: isActive ? '#007bff' : (isCompleted ? '#28a745' : '#f8f9fa'),
                  color: isActive ? 'white' : (isCompleted ? 'white' : '#6c757d'),
                  fontWeight: isActive ? 'bold' : 'normal',
                  transition: 'all 0.3s ease'
                }}
              >
                {stage.label}
              </div>
              {index < stages.length - 1 && (
                <div style={{
                  width: '30px',
                  height: '2px',
                  backgroundColor: isCompleted ? '#28a745' : '#ddd',
                  margin: '0 10px'
                }} />
              )}
            </div>
          );
        })}
      </div>
      
      {/* Eval mode indicator */}
      <div style={{
        position: 'absolute',
        right: '40px',
        padding: '6px 12px',
        backgroundColor: '#fff3cd',
        color: '#856404',
        borderRadius: '4px',
        fontSize: '12px',
        fontWeight: '500',
        border: '1px solid #ffeeba'
      }}>
        EVAL MODE 
      </div>
    </div>
  );
}

export default ProgressBar;
