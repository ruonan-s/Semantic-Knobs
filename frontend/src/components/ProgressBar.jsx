import React from 'react';

function ProgressBar({ currentStage }) {
  // Don't show progress bar for landing, upload, and input stages
  if (currentStage === 'landing' || currentStage === 'upload' || currentStage === 'input') {
    return null;
  }

  const stages = [
    'impression', 'impression_refinement'
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
          const isActive = currentStage === stage;
          const isCompleted = stages.indexOf(currentStage) > index;
          
          return (
            <div key={stage} style={{ display: 'flex', alignItems: 'center' }}>
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
                {stage === 'impression' ? 'Exploration' : 'Refinement'}
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
    </div>
  );
}

export default ProgressBar;