import React, { useState, useEffect, useCallback } from 'react';

/**
 * HITL Refinement Panel
 * 
 * Displays 4 images in a 2x2 grid with ordinal ranking (1st, 2nd, 3rd, 4th).
 * Users rank images to refine their aesthetic preferences through a GP-based
 * preference learning system.
 */
function HITLRefinementPanel({
  sessionId,
  round,
  images,           // Array of { id, url, filename }
  gpVariance,       // Current image variance (lower = images more similar = converging)
  isConverged,      // Whether the GP has converged
  isLoading,        // Loading state for image generation
  onSubmitRanking,  // Callback when user submits ranking
  onFinalize,       // Callback when user finalizes refinement
  onRollback,       // Callback when user clicks a previous best pick to rollback
  bestPicks,        // Array of { round, url, tags } - 1st-ranked images from previous rounds
  statusMessage     // Optional status message to display
}) {
  // Ranking state: maps rank (1-4) to image index (0-3)
  const [ranking, setRanking] = useState({ 1: null, 2: null, 3: null, 4: null });
  const [imagesLoaded, setImagesLoaded] = useState({});
  
  // Reset ranking when images change (new round)
  useEffect(() => {
    setRanking({ 1: null, 2: null, 3: null, 4: null });
    setImagesLoaded({});
  }, [images]);
  
  // Check if all images are ranked
  const allRanked = Object.values(ranking).every(v => v !== null);
  
  // Get the rank assigned to a specific image index
  const getImageRank = (imageIndex) => {
    for (const [rank, idx] of Object.entries(ranking)) {
      if (idx === imageIndex) return parseInt(rank);
    }
    return null;
  };
  
  // Handle rank button click
  const handleRankSelect = useCallback((imageIndex, rank) => {
    setRanking(prev => {
      const newRanking = { ...prev };
      
      // If this image already has a rank, clear it
      for (const [r, idx] of Object.entries(newRanking)) {
        if (idx === imageIndex) {
          newRanking[r] = null;
        }
      }
      
      // If this rank is already assigned to another image, swap
      const currentHolder = newRanking[rank];
      if (currentHolder !== null && currentHolder !== imageIndex) {
        // Find what rank the clicked image had
        const clickedImagePrevRank = Object.entries(prev).find(([r, idx]) => idx === imageIndex)?.[0];
        if (clickedImagePrevRank) {
          newRanking[clickedImagePrevRank] = currentHolder;
        }
      }
      
      // Assign the new rank
      newRanking[rank] = imageIndex;
      
      return newRanking;
    });
  }, []);
  
  // Handle submit
  const handleSubmit = useCallback(() => {
    if (!allRanked) return;
    
    // Convert ranking to array format expected by backend
    // ranking[1] = 2 means image at index 2 is ranked 1st
    // We need to produce [rank of image 0, rank of image 1, rank of image 2, rank of image 3]
    const rankingArray = [];
    for (let i = 0; i < 4; i++) {
      const rank = getImageRank(i);
      rankingArray.push(rank - 1); // Convert to 0-indexed
    }
    
    onSubmitRanking(rankingArray);
  }, [allRanked, ranking, onSubmitRanking]);
  
  // Rank colors
  const rankColors = {
    1: '#22c55e', // green - 1st
    2: '#84cc16', // lime - 2nd
    3: '#facc15', // yellow - 3rd
    4: '#f97316'  // orange - 4th
  };
  
  const rankLabels = {
    1: '1st',
    2: '2nd', 
    3: '3rd',
    4: '4th'
  };
  
  return (
    <div style={{ padding: '20px' }}>
      {/* Header */}
      <div style={{ 
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <h2 style={{ margin: 0, color: '#333' }}>
            Preference Refinement - Round {round}
          </h2>
          <p style={{ color: '#666', margin: '5px 0 0 0' }}>
            Rank the images from most preferred (1st) to least preferred (4th)
          </p>
        </div>
        
      </div>
      
      {/* Best Picks Gallery - Previous round winners */}
      {bestPicks && bestPicks.length > 0 && !isLoading && (
        <div style={{
          marginBottom: '20px',
          padding: '15px',
          backgroundColor: '#f8f9fa',
          borderRadius: '10px',
          border: '1px solid #e9ecef'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '10px'
          }}>
            <span style={{ fontSize: '14px', fontWeight: '600', color: '#495057' }}>
              Previous Best Picks
            </span>
            <span style={{ fontSize: '12px', color: '#868e96' }}>
              Not satisfied? Click a previous winner to go back to that round
            </span>
          </div>
          <div style={{
            display: 'flex',
            gap: '10px',
            overflowX: 'auto',
            paddingBottom: '5px'
          }}>
            {bestPicks.map((pick) => (
              <div
                key={`pick-${pick.round}`}
                onClick={() => onRollback && onRollback(pick.round)}
                style={{
                  flex: '0 0 auto',
                  width: '100px',
                  cursor: 'pointer',
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #dee2e6',
                  transition: 'all 0.2s ease',
                  position: 'relative',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = '#007bff';
                  e.currentTarget.style.transform = 'scale(1.05)';
                  e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,123,255,0.3)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = '#dee2e6';
                  e.currentTarget.style.transform = 'scale(1)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{
                  width: '100px',
                  height: '100px',
                  backgroundColor: '#e9ecef'
                }}>
                  {pick.url && (
                    <img
                      src={pick.url}
                      alt={`Round ${pick.round} winner`}
                      style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover'
                      }}
                    />
                  )}
                </div>
                <div style={{
                  padding: '4px 6px',
                  backgroundColor: '#fff',
                  textAlign: 'center',
                  fontSize: '11px',
                  color: '#495057',
                  fontWeight: '600',
                }}>
                  R{pick.round}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Status message */}
      {statusMessage && (
        <div style={{
          padding: '10px 15px',
          marginBottom: '20px',
          backgroundColor: '#fff3cd',
          border: '1px solid #ffeeba',
          borderRadius: '6px',
          color: '#856404',
          fontSize: '14px'
        }}>
          {statusMessage}
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
            borderTop: '4px solid #007bff',
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
          {/* 4 Images in One Row (like evaluation stage) */}
          <div style={{ 
            display: 'flex', 
            gap: '20px', 
            marginBottom: '20px',
            alignItems: 'flex-start'
          }}>
            {images.map((image, idx) => {
              const imageRank = getImageRank(idx);
              
              return (
                <div
                  key={image.id || idx}
                  style={{
                    flex: '1 1 0',
                    display: 'flex',
                    flexDirection: 'column',
                    backgroundColor: '#fff',
                    borderRadius: '12px',
                    border: imageRank ? `3px solid ${rankColors[imageRank]}` : '1px solid #dee2e6',
                    overflow: 'hidden',
                    maxWidth: '280px',
                    minWidth: 0,
                    boxShadow: imageRank ? `0 4px 12px ${rankColors[imageRank]}40` : '0 2px 8px rgba(0,0,0,0.1)',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {/* Image container */}
                  <div style={{
                    position: 'relative',
                    width: '100%',
                    paddingBottom: '100%',
                    backgroundColor: '#f0f0f0'
                  }}>
                    <img
                      src={image.url}
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
                        transform: 'translate(-50%, -50%)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '10px'
                      }}>
                        <div style={{
                          width: '30px',
                          height: '30px',
                          border: '3px solid #ddd',
                          borderTop: '3px solid #007bff',
                          borderRadius: '50%',
                          animation: 'spin 1s linear infinite'
                        }} />
                        <span style={{ fontSize: '12px', color: '#666' }}>Loading...</span>
                      </div>
                    )}
                    
                    {/* Rank badge */}
                    {imageRank && imagesLoaded[idx] && (
                      <div style={{
                        position: 'absolute',
                        top: '12px',
                        left: '12px',
                        width: '48px',
                        height: '48px',
                        backgroundColor: rankColors[imageRank],
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'white',
                        fontSize: '16px',
                        fontWeight: 'bold',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.2)'
                      }}>
                        {rankLabels[imageRank]}
                      </div>
                    )}
                  </div>
                  
                  {/* Rank buttons */}
                  <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    gap: '10px',
                    padding: '15px',
                    backgroundColor: '#f8f9fa'
                  }}>
                    {[1, 2, 3, 4].map(rank => {
                      const isSelected = ranking[rank] === idx;
                      const isAssigned = ranking[rank] !== null && ranking[rank] !== idx;
                      
                      return (
                        <button
                          key={rank}
                          onClick={() => handleRankSelect(idx, rank)}
                          style={{
                            width: '52px',
                            height: '52px',
                            borderRadius: '50%',
                            border: isSelected 
                              ? `3px solid ${rankColors[rank]}`
                              : '2px solid #dee2e6',
                            backgroundColor: isSelected ? rankColors[rank] : '#fff',
                            color: isSelected ? '#fff' : (isAssigned ? '#ccc' : '#333'),
                            fontSize: '14px',
                            fontWeight: 'bold',
                            cursor: 'pointer',
                            transition: 'all 0.2s ease',
                            opacity: isAssigned ? 0.5 : 1
                          }}
                          onMouseEnter={(e) => {
                            if (!isSelected) {
                              e.currentTarget.style.borderColor = rankColors[rank];
                              e.currentTarget.style.transform = 'scale(1.05)';
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (!isSelected) {
                              e.currentTarget.style.borderColor = '#dee2e6';
                              e.currentTarget.style.transform = 'scale(1)';
                            }
                          }}
                        >
                          {rankLabels[rank]}
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
          
          {/* Action buttons */}
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '20px',
            marginTop: '20px'
          }}>
            {/* Submit Ranking & Continue button */}
            <button
              onClick={handleSubmit}
              disabled={!allRanked}
              style={{
                padding: '14px 40px',
                fontSize: '16px',
                fontWeight: '600',
                backgroundColor: allRanked ? '#007bff' : '#ccc',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: allRanked ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s ease',
                boxShadow: allRanked ? '0 4px 12px rgba(0, 123, 255, 0.3)' : 'none'
              }}
              onMouseEnter={(e) => {
                if (allRanked) {
                  e.currentTarget.style.backgroundColor = '#0056b3';
                }
              }}
              onMouseLeave={(e) => {
                if (allRanked) {
                  e.currentTarget.style.backgroundColor = '#007bff';
                }
              }}
            >
              Submit Ranking & Continue →
            </button>
            
            {/* Finalize button - always visible so user can stop anytime */}
            <button
              onClick={onFinalize}
              disabled={round < 1}
              style={{
                padding: '14px 40px',
                fontSize: '16px',
                fontWeight: '600',
                backgroundColor: '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: '0 4px 12px rgba(40, 167, 69, 0.3)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#218838';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#28a745';
              }}
            >
              Finish & Save Preferences ✓
            </button>
          </div>
          
          {/* Round progress hint */}
          <p style={{
            textAlign: 'center',
            color: '#666',
            fontSize: '13px',
            marginTop: '15px'
          }}>
            {!allRanked 
              ? 'Click rank buttons below each image to assign rankings (1st to 4th)'
              : `Round ${round} complete • Click "Continue" for more refinement, or "Finish" to proceed to evaluation`
            }
          </p>
        </>
      )}
    </div>
  );
}

export default HITLRefinementPanel;
