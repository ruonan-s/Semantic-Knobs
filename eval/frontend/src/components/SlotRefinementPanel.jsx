import React, { useState, useEffect, useCallback } from 'react';

/**
 * Slot-Based Refinement Panel
 * 
 * Displays 4 images for slot-based elimination or weight refinement.
 * - Elimination stage: Head-to-head competition within semantic slots
 * - Refinement stage: Weight optimization for final tags
 */
function SlotRefinementPanel({
  sessionId,
  round,
  stage,              // 'elimination' or 'refinement' or 'weight_refinement'
  roundType,          // 'exploration', 'head_to_head', 'validation', 'weight_optimization'
  focusSlot,          // Current slot being compared (for head-to-head)
  images,             // Array of { url, filename }
  compositions,       // For elimination: slot selections per image
  weightConfigs,      // For refinement: weight configs per image
  slotsStatus,        // Array of { name, winner, confidence, is_resolved, remaining_tags }
  currentWeights,     // Current tag weights (for refinement stage)
  isLoading,
  onSubmitSelection,  // Callback when user selects an image
  onFinalize,
  statusMessage
}) {
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [imagesLoaded, setImagesLoaded] = useState({});
  
  // Reset selection when images change
  useEffect(() => {
    setSelectedIdx(null);
    setImagesLoaded({});
  }, [images]);
  
  // Handle image selection
  const handleSelect = useCallback((idx) => {
    setSelectedIdx(idx);
  }, []);
  
  // Handle submit
  const handleSubmit = useCallback(() => {
    if (selectedIdx === null) return;
    onSubmitSelection(selectedIdx);
    setSelectedIdx(null);
  }, [selectedIdx, onSubmitSelection]);
  
  // Calculate progress for elimination
  const resolvedSlots = slotsStatus?.filter(s => s.is_resolved).length || 0;
  const totalSlots = slotsStatus?.length || 0;
  const progressPercent = totalSlots > 0 ? (resolvedSlots / totalSlots) * 100 : 0;
  
  // Get stage display info
  const getStageInfo = () => {
    if (stage === 'elimination') {
      if (roundType === 'exploration') {
        return {
          title: 'Exploration Round',
          subtitle: 'Select the image that best matches your preferences',
          color: '#6366f1'
        };
      } else if (roundType === 'head_to_head') {
        return {
          title: `Comparing: ${focusSlot?.replace(/_/g, ' ')}`,
          subtitle: 'Which variation do you prefer for this design element?',
          color: '#8b5cf6'
        };
      } else {
        return {
          title: 'Validation Round',
          subtitle: 'Confirm your final selections',
          color: '#10b981'
        };
      }
    } else {
      return {
        title: 'Weight Refinement',
        subtitle: 'Which balance of elements looks best?',
        color: '#f59e0b'
      };
    }
  };
  
  const stageInfo = getStageInfo();
  
  return (
    <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto' }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'flex-start'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
            <span style={{
              backgroundColor: stageInfo.color,
              color: 'white',
              padding: '4px 12px',
              borderRadius: '16px',
              fontSize: '12px',
              fontWeight: '600',
              textTransform: 'uppercase'
            }}>
              {stage === 'elimination' ? 'Elimination' : 'Refinement'}
            </span>
            <span style={{ color: '#666', fontSize: '14px' }}>
              Round {round}
            </span>
          </div>
          
          <h2 style={{ margin: 0, color: '#333', fontSize: '24px' }}>
            {stageInfo.title}
          </h2>
          <p style={{ color: '#666', margin: '8px 0 0 0', fontSize: '15px' }}>
            {stageInfo.subtitle}
          </p>
        </div>
        
        {/* Progress indicator for elimination */}
        {stage === 'elimination' && slotsStatus && (
          <div style={{
            backgroundColor: '#f8f9fa',
            padding: '12px 20px',
            borderRadius: '8px',
            minWidth: '200px'
          }}>
            <div style={{ fontSize: '13px', color: '#666', marginBottom: '6px' }}>
              Slots Resolved
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                flex: 1,
                height: '8px',
                backgroundColor: '#e9ecef',
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{
                  width: `${progressPercent}%`,
                  height: '100%',
                  backgroundColor: '#10b981',
                  transition: 'width 0.3s ease'
                }} />
              </div>
              <span style={{ fontSize: '14px', fontWeight: '600', color: '#333' }}>
                {resolvedSlots}/{totalSlots}
              </span>
            </div>
          </div>
        )}
      </div>
      
      {/* Status message */}
      {statusMessage && (
        <div style={{
          padding: '12px 16px',
          marginBottom: '20px',
          backgroundColor: '#e8f4fd',
          border: '1px solid #bde0fe',
          borderRadius: '8px',
          color: '#1e40af',
          fontSize: '14px'
        }}>
          {statusMessage}
        </div>
      )}
      
      {/* Slot status chips (for elimination) */}
      {stage === 'elimination' && slotsStatus && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px',
          marginBottom: '20px'
        }}>
          {slotsStatus.map((slot, idx) => (
            <div
              key={slot.name}
              style={{
                padding: '6px 12px',
                borderRadius: '20px',
                fontSize: '12px',
                fontWeight: '500',
                backgroundColor: slot.is_resolved ? '#dcfce7' : 
                               (slot.name === focusSlot ? '#dbeafe' : '#f3f4f6'),
                color: slot.is_resolved ? '#166534' : 
                       (slot.name === focusSlot ? '#1e40af' : '#6b7280'),
                border: slot.name === focusSlot ? '2px solid #3b82f6' : 'none'
              }}
            >
              {slot.name.replace(/_/g, ' ')}
              {slot.is_resolved && ' ✓'}
              {slot.winner && !slot.is_resolved && `: ${slot.winner}`}
            </div>
          ))}
        </div>
      )}
      
      {/* Loading state */}
      {isLoading ? (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '80px 20px',
          backgroundColor: '#f8f9fa',
          borderRadius: '12px'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid #e9ecef',
            borderTop: `4px solid ${stageInfo.color}`,
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <p style={{ marginTop: '20px', color: '#666', fontSize: '16px' }}>
            Generating images for round {round}...
          </p>
          <style>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      ) : (
        <>
          {/* 4 Images in a row */}
          <div style={{ 
            display: 'flex', 
            gap: '16px', 
            marginBottom: '20px',
            alignItems: 'stretch'
          }}>
            {images && images.map((image, idx) => {
              const isSelected = selectedIdx === idx;
              const composition = compositions?.[idx];
              const weights = weightConfigs?.[idx];
              
              return (
                <div
                  key={idx}
                  onClick={() => handleSelect(idx)}
                  style={{
                    flex: '1 1 0',
                    display: 'flex',
                    flexDirection: 'column',
                    backgroundColor: '#fff',
                    borderRadius: '12px',
                    border: isSelected 
                      ? `3px solid ${stageInfo.color}` 
                      : '2px solid #e5e7eb',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    transform: isSelected ? 'scale(1.02)' : 'scale(1)',
                    boxShadow: isSelected 
                      ? `0 8px 24px ${stageInfo.color}30` 
                      : '0 2px 8px rgba(0,0,0,0.08)'
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.borderColor = stageInfo.color;
                      e.currentTarget.style.boxShadow = `0 4px 16px ${stageInfo.color}20`;
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) {
                      e.currentTarget.style.borderColor = '#e5e7eb';
                      e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
                    }
                  }}
                >
                  {/* Image */}
                  <div style={{
                    position: 'relative',
                    width: '100%',
                    paddingBottom: '100%',
                    backgroundColor: '#f0f0f0'
                  }}>
                    <img
                      src={image.url || image}
                      alt={`Option ${idx + 1}`}
                      style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        opacity: imagesLoaded[idx] ? 1 : 0,
                        transition: 'opacity 0.3s ease'
                      }}
                      onLoad={() => setImagesLoaded(prev => ({ ...prev, [idx]: true }))}
                    />
                    
                    {/* Loading spinner */}
                    {!imagesLoaded[idx] && (
                      <div style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)'
                      }}>
                        <div style={{
                          width: '30px',
                          height: '30px',
                          border: '3px solid #ddd',
                          borderTop: `3px solid ${stageInfo.color}`,
                          borderRadius: '50%',
                          animation: 'spin 1s linear infinite'
                        }} />
                      </div>
                    )}
                    
                    {/* Selection indicator */}
                    {isSelected && imagesLoaded[idx] && (
                      <div style={{
                        position: 'absolute',
                        top: '10px',
                        right: '10px',
                        width: '36px',
                        height: '36px',
                        backgroundColor: stageInfo.color,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: '20px',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                      }}>
                        ✓
                      </div>
                    )}
                  </div>
                  
                  {/* Info panel */}
                  <div style={{
                    padding: '12px',
                    backgroundColor: isSelected ? `${stageInfo.color}10` : '#f9fafb',
                    borderTop: '1px solid #e5e7eb',
                    fontSize: '12px',
                    color: '#666'
                  }}>
                    {/* For elimination: show slot selection */}
                    {composition && (
                      <div>
                        {focusSlot && composition.slot_selections?.[focusSlot] && (
                          <div style={{ 
                            fontWeight: '600', 
                            color: '#333',
                            marginBottom: '4px'
                          }}>
                            {composition.slot_selections[focusSlot]}
                          </div>
                        )}
                        <div style={{ color: '#999', fontSize: '11px' }}>
                          {composition.strategy?.replace(/_/g, ' ')}
                        </div>
                      </div>
                    )}
                    
                    {/* For refinement: show weight strategy */}
                    {weights && (
                      <div style={{ color: '#666' }}>
                        {weights.strategy?.replace(/_/g, ' ')}
                      </div>
                    )}
                    
                    {!composition && !weights && (
                      <div style={{ color: '#999' }}>Option {idx + 1}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Action buttons */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '16px',
            marginTop: '24px'
          }}>
            <button
              onClick={handleSubmit}
              disabled={selectedIdx === null}
              style={{
                padding: '14px 48px',
                fontSize: '16px',
                fontWeight: '600',
                backgroundColor: selectedIdx !== null ? stageInfo.color : '#d1d5db',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: selectedIdx !== null ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s ease',
                boxShadow: selectedIdx !== null ? `0 4px 12px ${stageInfo.color}40` : 'none'
              }}
              onMouseEnter={(e) => {
                if (selectedIdx !== null) {
                  e.currentTarget.style.transform = 'translateY(-1px)';
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              Select & Continue
            </button>
            
            {/* Finalize button */}
            {round >= 3 && (
              <button
                onClick={onFinalize}
                style={{
                  padding: '14px 32px',
                  fontSize: '16px',
                  fontWeight: '600',
                  backgroundColor: '#6b7280',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = '#4b5563';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = '#6b7280';
                }}
              >
                Finish & Save
              </button>
            )}
          </div>
          
          {/* Selection hint */}
          {selectedIdx === null && (
            <p style={{
              textAlign: 'center',
              color: '#9ca3af',
              fontSize: '14px',
              marginTop: '12px'
            }}>
              Click on an image to select it
            </p>
          )}
        </>
      )}
      
      {/* Current weights display (for refinement) */}
      {stage === 'weight_refinement' && currentWeights && (
        <div style={{
          marginTop: '24px',
          padding: '16px',
          backgroundColor: '#fffbeb',
          borderRadius: '8px',
          border: '1px solid #fde68a'
        }}>
          <div style={{ fontSize: '13px', fontWeight: '600', color: '#92400e', marginBottom: '12px' }}>
            Current Tag Weights
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {Object.entries(currentWeights)
              .sort(([, a], [, b]) => b - a)
              .map(([tag, weight]) => (
                <div
                  key={tag}
                  style={{
                    padding: '4px 10px',
                    backgroundColor: 'white',
                    borderRadius: '4px',
                    fontSize: '12px',
                    border: '1px solid #fde68a'
                  }}
                >
                  <span style={{ color: '#333' }}>{tag}</span>
                  <span style={{ 
                    marginLeft: '6px', 
                    color: weight >= 1 ? '#059669' : '#6b7280',
                    fontWeight: '600'
                  }}>
                    {weight.toFixed(2)}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default SlotRefinementPanel;
