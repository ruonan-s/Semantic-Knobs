import React, { useState, useEffect } from 'react';

/**
 * StageSelector component for choosing which stage to refine
 * Used in Test Stage Refinement workflow
 */
function StageSelector({ sessionPath, sessionName, onStageSelect, onBack }) {
  const [stagesInfo, setStagesInfo] = useState([]);
  const [preferences, setPreferences] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedStage, setSelectedStage] = useState(null);

  useEffect(() => {
    loadSession();
  }, [sessionPath]);

  const loadSession = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/load-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_path: sessionPath })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to load session: ${response.status}`);
      }
      
      const data = await response.json();
      setStagesInfo(data.stages || []);
      setPreferences(data.preferences || {});
      setError(null);
    } catch (err) {
      console.error('Error loading session:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStageClick = (stage) => {
    if (stage.can_refine) {
      setSelectedStage(stage.stage_name);
    }
  };

  const handleConfirmSelection = () => {
    if (selectedStage) {
      onStageSelect(selectedStage);
    }
  };

  const getStageDisplayName = (stageName) => {
    const names = {
      impression: 'Impression',
      spatial: 'Spatial Layout',
      objects: 'Objects & Materials',
      ambient: 'Ambient & Lighting'
    };
    return names[stageName] || stageName;
  };

  const getStageDescription = (stageName) => {
    const descriptions = {
      impression: 'Overall aesthetic and design philosophy',
      spatial: 'Physical structure and spatial organization',
      objects: 'Furniture, fixtures, and material choices',
      ambient: 'Lighting, atmosphere, and mood'
    };
    return descriptions[stageName] || '';
  };

  const getStageIcon = (stageName) => {
    const icons = {
      impression: '🎨',
      spatial: '🏗️',
      objects: '🪑',
      ambient: '💡'
    };
    return icons[stageName] || '📐';
  };

  if (loading) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        color: '#666'
      }}>
        <div style={{ fontSize: '18px', marginBottom: '10px' }}>Loading session data...</div>
        <div style={{ fontSize: '14px' }}>Please wait</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center'
      }}>
        <div style={{ color: '#f44336', fontSize: '18px', marginBottom: '20px' }}>
          Error loading session
        </div>
        <div style={{ color: '#666', marginBottom: '20px' }}>
          {error}
        </div>
        <button
          onClick={onBack}
          style={{
            padding: '10px 20px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          ← Back to Sessions
        </button>
      </div>
    );
  }

  return (
    <div style={{
      padding: '30px',
      maxWidth: '1000px',
      margin: '0 auto'
    }}>
      <div style={{
        marginBottom: '30px'
      }}>
        <button
          onClick={onBack}
          style={{
            padding: '8px 16px',
            backgroundColor: '#f5f5f5',
            color: '#666',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px',
            marginBottom: '20px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          ← Back
        </button>
        
        <h2 style={{
          fontSize: '24px',
          fontWeight: '600',
          color: '#333',
          marginBottom: '8px'
        }}>
          Select Starting Stage
        </h2>
        <p style={{
          color: '#666',
          fontSize: '14px',
          marginBottom: '8px'
        }}>
          Session: <strong>{sessionName}</strong>
        </p>
        <p style={{
          color: '#999',
          fontSize: '13px'
        }}>
          Choose which stage to refine. You can manipulate concepts and generate refined images.
        </p>
      </div>

      {/* Stages Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
        gap: '20px',
        marginBottom: '30px'
      }}>
        {stagesInfo.map((stage) => {
          const isSelected = selectedStage === stage.stage_name;
          const canRefine = stage.can_refine;
          
          return (
            <div
              key={stage.stage_name}
              onClick={() => handleStageClick(stage)}
              style={{
                padding: '20px',
                border: '2px solid',
                borderColor: isSelected ? '#2196F3' : '#e0e0e0',
                borderRadius: '12px',
                cursor: canRefine ? 'pointer' : 'not-allowed',
                backgroundColor: isSelected ? '#e3f2fd' : canRefine ? 'white' : '#f5f5f5',
                opacity: canRefine ? 1 : 0.6,
                transition: 'all 0.2s',
                position: 'relative'
              }}
              onMouseEnter={(e) => {
                if (canRefine && !isSelected) {
                  e.currentTarget.style.borderColor = '#90caf9';
                  e.currentTarget.style.backgroundColor = '#f5f5f5';
                }
              }}
              onMouseLeave={(e) => {
                if (canRefine && !isSelected) {
                  e.currentTarget.style.borderColor = '#e0e0e0';
                  e.currentTarget.style.backgroundColor = 'white';
                }
              }}
            >
              {isSelected && (
                <div style={{
                  position: 'absolute',
                  top: '10px',
                  right: '10px',
                  width: '24px',
                  height: '24px',
                  borderRadius: '50%',
                  backgroundColor: '#2196F3',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontSize: '12px',
                  fontWeight: 'bold'
                }}>
                  ✓
                </div>
              )}
              
              <div style={{
                fontSize: '36px',
                marginBottom: '12px'
              }}>
                {getStageIcon(stage.stage_name)}
              </div>
              
              <div style={{
                fontSize: '16px',
                fontWeight: '600',
                color: '#333',
                marginBottom: '8px'
              }}>
                {getStageDisplayName(stage.stage_name)}
              </div>
              
              <div style={{
                fontSize: '12px',
                color: '#666',
                marginBottom: '12px',
                lineHeight: '1.4'
              }}>
                {getStageDescription(stage.stage_name)}
              </div>
              
              <div style={{
                fontSize: '11px',
                color: '#999',
                borderTop: '1px solid #e0e0e0',
                paddingTop: '10px'
              }}>
                {stage.has_images && (
                  <div>📷 {stage.image_count} images</div>
                )}
                {stage.has_tags && (
                  <div>🏷️ Tags available</div>
                )}
                {stage.has_refinement && (
                  <div style={{ color: '#4caf50' }}>✓ Already refined</div>
                )}
                {!canRefine && (
                  <div style={{ color: '#f44336' }}>❌ Cannot refine</div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {stagesInfo.length === 0 && (
        <div style={{
          padding: '40px',
          textAlign: 'center',
          color: '#999',
          border: '2px dashed #e0e0e0',
          borderRadius: '12px'
        }}>
          No stages found in this session
        </div>
      )}

      {/* Action Buttons */}
      <div style={{
        display: 'flex',
        gap: '12px',
        justifyContent: 'flex-end',
        marginTop: '30px'
      }}>
        <button
          onClick={onBack}
          style={{
            padding: '12px 24px',
            backgroundColor: '#f5f5f5',
            color: '#666',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.2s'
          }}
        >
          Cancel
        </button>
        
        <button
          onClick={handleConfirmSelection}
          disabled={!selectedStage}
          style={{
            padding: '12px 24px',
            backgroundColor: selectedStage ? '#2196F3' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: selectedStage ? 'pointer' : 'not-allowed',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.2s'
          }}
        >
          Load Stage →
        </button>
      </div>

      {/* Info Box */}
      <div style={{
        marginTop: '30px',
        padding: '16px',
        backgroundColor: '#fff3e0',
        border: '1px solid #ffe0b2',
        borderRadius: '8px',
        fontSize: '13px',
        color: '#555'
      }}>
        <strong style={{ color: '#f57c00' }}>⚡ How it works:</strong> After selecting a stage, 
        you'll see its existing images and tags. Like/dislike tags to build your concept preferences, 
        then generate 4 refined images that converge toward your preferences.
      </div>
    </div>
  );
}

export default StageSelector;

