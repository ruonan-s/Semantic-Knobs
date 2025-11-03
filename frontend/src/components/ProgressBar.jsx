import React from 'react';

function ProgressBar({ currentStage }) {
  // Don't show progress bar for landing and upload stages
  if (currentStage === 'landing' || currentStage === 'upload') {
    return null;
  }
  
  const stages = [
    'impression', 'impression_refinement',
    'spatial', 'spatial_refinement',
    'objects', 'objects_refinement',
    'ambient', 'ambient_refinement',
    'mode-selection', 'final'
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
                {stage === 'mode-selection' ? 'Mode Selection' : 
                 stage.includes('_refinement') ? stage.replace('_refinement', '').charAt(0).toUpperCase() + stage.replace('_refinement', '').slice(1) + ' ↻' :
                 stage.charAt(0).toUpperCase() + stage.slice(1)}
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