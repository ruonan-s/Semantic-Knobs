/**
 * RefinementIterationControls - Multi-round PBO refinement
 *
 * CORRECT LOGIC:
 * - Reference image is FIXED (always the original impression selection)
 * - Each refinement image represents a weight vector over concepts
 * - Selecting an image = selecting weights = preference signal
 * - "Refine More" automatically records selection + proposes + generates
 * - All rounds use the SAME reference image with different fused embeddings
 */

import React, { useState, useEffect, useCallback } from 'react';

const RefinementIterationControls = ({
  sessionId,
  stage,  // base stage (e.g., "impression", not "impression_refinement")
  images,
  selectedImage,
  onRefinementComplete,
  onContinue,
  disabled = false,
  initialRound = 1
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState(null);
  const [round, setRound] = useState(initialRound);
  const [selectionHistory, setSelectionHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  
  // Historical selection state - when user clicks a previous selection
  const [selectedHistoricalItem, setSelectedHistoricalItem] = useState(null);
  
  // Tag injection state
  const [injectedTag, setInjectedTag] = useState('');
  const [injectedEmphasis, setInjectedEmphasis] = useState('mid');

  // Load selection history when component mounts or round changes
  const loadSelectionHistory = useCallback(async () => {
    console.log('[History] ===== LOADING SELECTION HISTORY =====');
    console.log('[History] sessionId:', sessionId);
    console.log('[History] stage:', stage);
    console.log('[History] current round:', round);
    
    try {
      setLoadingHistory(true);
      
      const history = [];
      
      // STEP 1: Always load reference image from preferences.json first
      // The reference is the selected image from the previous stage (e.g., impression)
      console.log('[History] Loading reference image from preferences...');
      try {
        const prefsResponse = await fetch(`/sessions/${sessionId}/preferences.json`);
        if (prefsResponse.ok) {
          const prefsData = await prefsResponse.json();
          const selectedImageId = prefsData.selections?.[stage];  // e.g., "impression_0_0"
          console.log('[History] Selected image ID from preferences:', selectedImageId);
          
          if (selectedImageId) {
            // Construct reference image URL
            const refImageUrl = `/sessions/${sessionId}/${stage}/${selectedImageId}.png`;
            const refItem = {
              type: 'reference',
              round: 0,
              imageId: selectedImageId,
              imageUrl: refImageUrl,
              label: 'Reference',
              weights: null  // Reference doesn't have PBO weights
            };
            history.push(refItem);
            console.log('[History] ✅ Added reference image from preferences:', refItem);
          } else {
            console.log('[History] ⚠️ No selected image found in preferences for stage:', stage);
          }
        } else {
          console.log('[History] ⚠️ Failed to load preferences.json');
        }
      } catch (err) {
        console.log('[History] ⚠️ Error loading preferences for reference:', err);
      }
      
      // STEP 2: Try to load tracking.json for round selections
      // Note: tracking.json is in the refinement stage folder (e.g., impression_refinement)
      const trackingUrl = `/sessions/${sessionId}/${stage}_refinement/tracking.json`;
      console.log('[History] Fetching tracking data from:', trackingUrl);
      const response = await fetch(trackingUrl);
      
      console.log('[History] Response status:', response.status, response.statusText);
      if (!response.ok) {
        if (response.status === 404) {
          console.log('[History] No tracking.json found yet (this is normal for Round 1)');
          console.log('[History] Showing only reference image');
        } else {
          console.log('[History] No tracking data yet - response not ok');
        }
        // Still show reference image even if no tracking data
        setSelectionHistory(history);
        return;
      }
      
      const trackingData = await response.json();
      console.log('[History] Loaded tracking data:', trackingData);
      console.log('[History] Current round state:', round);
      console.log('[History] Total rounds in tracking:', trackingData.rounds?.length || 0);
      
      // Auto-detect current round: it's the last round in tracking (the one we're viewing)
      // The tracking.json already includes the current round with its proposals
      // OR if there are no rounds yet, it's round 1
      const actualCurrentRound = trackingData.rounds && trackingData.rounds.length > 0
        ? Math.max(...trackingData.rounds.map(r => r.round_number))
        : 1;
      
      console.log('[History] Auto-detected actual current round:', actualCurrentRound);
      
      // Update round state if it's out of sync
      if (actualCurrentRound !== round) {
        console.log(`[History] Syncing round state from ${round} to ${actualCurrentRound}`);
        setRound(actualCurrentRound);
      }
      
      // Add selected images from previous rounds (not current round)
      // Current round images are already shown in the main 2x2 grid
      for (const roundData of trackingData.rounds || []) {
        const isPreviousRound = roundData.round_number < actualCurrentRound;
        console.log(`[History] Checking round ${roundData.round_number}:`, {
          has_selection: !!roundData.user_selection,
          is_previous_round: isPreviousRound,
          user_selection: roundData.user_selection,
          proposals_count: roundData.proposals?.length || 0
        });
        
        // Only include rounds with selections that are NOT the current round
        if (roundData.user_selection && isPreviousRound) {
          const selectedIdx = roundData.user_selection.selected_index;
          
          // Load weights for this round
          const weightsResponse = await fetch(`/sessions/${sessionId}/${stage}_refinement/round_${roundData.round_number}/weights.json`);
          let weights = null;
          
          if (weightsResponse.ok) {
            const weightsData = await weightsResponse.json();
            // Get the weight vector for the selected image
            weights = weightsData.proposals[selectedIdx];
            console.log(`[History] Loaded weights for round ${roundData.round_number}, image ${selectedIdx}`);
          } else {
            console.log(`[History] Failed to load weights for round ${roundData.round_number}`);
          }
          
          history.push({
            type: 'selection',
            round: roundData.round_number,
            imageId: `round_${roundData.round_number}_image_${selectedIdx}`,
            imageUrl: `/sessions/${sessionId}/${stage}_refinement/round_${roundData.round_number}/image_${selectedIdx}.png`,
            label: `Round ${roundData.round_number}`,
            weights: weights
          });
        }
      }
      
      console.log(`[History] Filtered to show only rounds < ${actualCurrentRound} (current round excluded)`);
    
      
      setSelectionHistory(history);
      console.log('[History] Loaded', history.length, 'selections');
      
    } catch (err) {
      console.error('[History] ===== ERROR LOADING HISTORY =====');
      console.error('[History] Error:', err);
      console.error('[History] Stack:', err.stack);
    } finally {
      setLoadingHistory(false);
      console.log('[History] ===== FINISHED LOADING HISTORY =====');
    }
  }, [sessionId, stage]);

  // Effect to load history when component mounts or round changes
  useEffect(() => {
    console.log('[History] useEffect triggered - calling loadSelectionHistory');
    loadSelectionHistory();
  }, [loadSelectionHistory, round]);

  // Check if we can proceed - either current selection OR historical selection
  const hasCurrentSelection = images && images.length === 4 && selectedImage !== null;
  const hasHistoricalSelection = selectedHistoricalItem !== null;
  const canRefineMore = (hasCurrentSelection || hasHistoricalSelection) && !isGenerating;

  // Handle clicking on a historical image - just select it, don't generate yet
  const handleHistoricalSelection = (historyItem) => {
    if (!historyItem.weights) {
      console.log('[History] Reference image has no weights, cannot use for refinement');
      setStatus(`⚠️ ${historyItem.label} has no weights - cannot refine from it`);
      setTimeout(() => setStatus(''), 3000);
      return;
    }
    
    // Toggle selection - if already selected, deselect
    if (selectedHistoricalItem?.imageId === historyItem.imageId) {
      console.log('[History] Deselecting historical item:', historyItem.label);
      setSelectedHistoricalItem(null);
      setStatus('');
    } else {
      console.log('[History] Selecting historical item:', historyItem.label);
      setSelectedHistoricalItem(historyItem);
      setStatus(`📌 Selected ${historyItem.label} - Click "Refine More" to generate new variations or "Save selection" to use it`);
    }
  };

  // Handle refining from historical selection
  const handleRefineFromHistory = async () => {
    if (!selectedHistoricalItem) return;
    
    try {
      setIsGenerating(true);
      setStatus(`🔄 Round ${round + 1}: Generating from ${selectedHistoricalItem.label}...`);
      setError(null);
      
      // Get current round image IDs for preference recording
      // (historical selection > all current round images)
      const currentRoundImageIds = images ? images.map(img => img.id) : [];
      console.log('[History] Recording preference: historical > current images:', currentRoundImageIds);
      
      // Generate new proposals using the historical weights
      const response = await fetch('/api/pbo/refine-from-weights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          weights: selectedHistoricalItem.weights,
          round_number: round,
          current_round_image_ids: currentRoundImageIds
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to refine from history: ${response.statusText}`);
      }
      
      const data = await response.json();
      const newRound = data.round_number;
      setRound(newRound);
      setSelectedHistoricalItem(null); // Clear historical selection after use
      setStatus(`✅ Round ${newRound} complete! Generated 4 new variations from ${selectedHistoricalItem.label}.`);
      
      // Notify parent component with new images
      if (onRefinementComplete) {
        const newImages = data.image_paths.map((path, i) => ({
          id: `round_${newRound}_image_${i}`,
          url: path
        }));
        onRefinementComplete(newImages, newRound);
      }
      
      // Reload history
      await loadSelectionHistory();
      
      // Auto-clear status after 3s
      setTimeout(() => setStatus(''), 3000);
      
    } catch (err) {
      console.error('Error refining from history:', err);
      setError(err.message);
      setStatus('');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRefineMore = async () => {
    if (!canRefineMore) return;

    // If a historical image is selected, use that flow instead
    if (selectedHistoricalItem) {
      return handleRefineFromHistory();
    }

    // Otherwise, refine from current round selection
    if (!hasCurrentSelection) return;

    try {
      setIsGenerating(true);
      setStatus(`🔄 Round ${round + 1}: Learning from your selection...`);
      setError(null);

      // Extract round number from the first image ID
      // e.g., "round_2_image_0" -> round 2
      // OR legacy format: "impression_refinement_0_0" -> round 1
      const firstImageId = images[0]?.id || '';
      const imageRoundMatch = firstImageId.match(/round_(\d+)_/);
      
      let actualImageRound;
      if (imageRoundMatch) {
        // New format: round_X_image_Y
        actualImageRound = parseInt(imageRoundMatch[1]);
      } else if (firstImageId.match(/^impression_refinement_\d+_\d+$/)) {
        // Legacy format from initial refinement: treat as Round 1
        actualImageRound = 1;
      } else {
        // Fallback to state
        actualImageRound = round;
      }
      
      console.log('[Refine More] Component round state:', round);
      console.log('[Refine More] First image ID:', firstImageId);
      console.log('[Refine More] Detected image round:', actualImageRound);
      console.log('[Refine More] Selected image:', selectedImage);

      const requestBody = {
        session_id: sessionId,
        stage: stage,
        selected_image_id: selectedImage,
        all_image_ids: images.map(img => img.id),
        round_number: actualImageRound  // Use the round from the images, not the state
      };
      
      // Add tag injection if user has entered a tag
      if (injectedTag && injectedTag.trim() !== '') {
        requestBody.injected_tag = injectedTag.trim();
        requestBody.injected_emphasis = injectedEmphasis;
        console.log('[Refine More] Including tag injection:', {
          tag: injectedTag.trim(),
          emphasis: injectedEmphasis
        });
      }
      
      console.log('[Refine More] Request body:', requestBody);

      // Call unified endpoint: record + propose + generate
      const response = await fetch('/api/pbo/refine-next-round', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      console.log('[Refine More] Response status:', response.status, response.statusText);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[Refine More] Error response:', errorText);
        throw new Error(`Failed to refine: ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      const newRound = data.round_number;
      setRound(newRound);
      setStatus(`✅ Round ${newRound} complete! PBO generated 4 new variations.`);

      // Clear injected tag after successful generation
      setInjectedTag('');
      console.log('[Refine More] Cleared injected tag after successful generation');

      // Notify parent component with new images
      if (onRefinementComplete) {
        const newImages = data.image_paths.map((path, i) => ({
          id: `round_${newRound}_image_${i}`,
          url: path
        }));
        onRefinementComplete(newImages, newRound);
      }

      // Reload history after new round
      await loadSelectionHistory();

      // Auto-clear status after 3s
      setTimeout(() => setStatus(''), 3000);

    } catch (err) {
      console.error('Error refining:', err);
      setError(err.message);
      setStatus('');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>
          🔄 Refinement Round {round}
        </h3>
        <p style={styles.description}>
          Select an image and choose: continue to next stage or refine more with PBO
        </p>
      </div>

      {/* Tag Injection Section */}
      <div style={styles.tagInjectionContainer}>
        <h4 style={styles.tagInjectionTitle}>🏷️ Tag Injection (Optional)</h4>
        <p style={styles.tagInjectionDescription}>
          Add a custom tag to influence the next round. Leave empty to skip.
        </p>
        <div style={styles.tagInjectionControls}>
          <input
            type="text"
            value={injectedTag}
            onChange={(e) => setInjectedTag(e.target.value)}
            placeholder="Enter custom tag (e.g., 'warm lighting')..."
            style={styles.tagInput}
            disabled={isGenerating}
          />
          <select
            value={injectedEmphasis}
            onChange={(e) => setInjectedEmphasis(e.target.value)}
            style={styles.emphasisSelect}
            disabled={isGenerating}
          >
            <option value="high">High (50%)</option>
            <option value="mid">Mid (30%)</option>
            <option value="low">Low (10%)</option>
          </select>
        </div>
        {injectedTag && injectedTag.trim() !== '' && (
          <div style={styles.tagPreview}>
            ✓ Will inject: <strong>"{injectedTag.trim()}"</strong> with <strong>{injectedEmphasis}</strong> emphasis
          </div>
        )}
      </div>

      <div style={styles.controls}>
        {/* Continue to Next Stage */}
        <button
          onClick={() => {
            if (selectedHistoricalItem) {
              // For historical selection, pass the historical item info to parent
              onContinue(selectedHistoricalItem);
            } else {
              onContinue();
            }
          }}
          disabled={(!selectedImage && !selectedHistoricalItem) || disabled || isGenerating}
          style={{
            ...styles.button,
            ...styles.continueButton,
            ...(((!selectedImage && !selectedHistoricalItem) || disabled || isGenerating) ? styles.buttonDisabled : {})
          }}
          title={selectedHistoricalItem 
            ? `Save ${selectedHistoricalItem.label} and proceed to next stage`
            : "Proceed to the next stage with this selection"
          }
        >
          {selectedHistoricalItem ? `Save ${selectedHistoricalItem.label} →` : 'Save selection →'}
        </button>

        {/* Refine More Button */}
        <button
          onClick={handleRefineMore}
          disabled={!canRefineMore || disabled}
          style={{
            ...styles.button,
            ...styles.refineButton,
            ...((!canRefineMore || disabled) ? styles.buttonDisabled : {})
          }}
          title={selectedHistoricalItem 
            ? `Generate variations from ${selectedHistoricalItem.label}`
            : "Generate 4 new variations based on your selection"
          }
        >
          {isGenerating 
            ? '⏳ Generating Round ' + (round + 1) + '...' 
            : (selectedHistoricalItem ? `🔄 Refine from ${selectedHistoricalItem.label}` : '🔄 Refine More')
          }
        </button>
      </div>

      {/* Status/Error Display */}
      {status && (
        <div style={styles.status}>
          {status}
        </div>
      )}

      {error && (
        <div style={styles.error}>
          ❌ {error}
        </div>
      )}

      {/* Info Box */}
      <div style={styles.infoBox}>
        <span>Round:</span>
        <span style={styles.badge}>{round}</span>
      </div>

      {/* Selection History */}
      <div style={styles.historyContainer}>
        <h4 style={styles.historyTitle}>
          📜 Selection History
        </h4>
        <p style={styles.historyDescription}>
          Click any previous selection to select it, then use "Refine More" or "Save selection"
        </p>
        
        {loadingHistory ? (
          <div style={styles.historyLoading}>Loading history...</div>
        ) : selectionHistory.length === 0 ? (
          <div style={styles.historyEmpty}>
            No selection history available yet.
            <br/>
            <span style={{fontSize: '12px', opacity: 0.7}}>
              History appears after completing refinement rounds.
            </span>
          </div>
        ) : (
          <div style={styles.historyGrid}>
            {selectionHistory.slice(0, 9).map((item, idx) => {
              const isSelected = selectedHistoricalItem?.imageId === item.imageId;
              return (
                <div
                  key={`${item.type}-${item.round}-${idx}`}
                  style={{
                    ...styles.historyItem,
                    ...(item.type === 'reference' ? styles.historyItemReference : {}),
                    ...(item.weights === null ? styles.historyItemDisabled : {}),
                    ...(isSelected ? styles.historyItemSelected : {})
                  }}
                  onClick={() => item.weights && handleHistoricalSelection(item)}
                  title={item.weights 
                    ? (isSelected ? `${item.label} selected - Click again to deselect` : `Click to select ${item.label}`)
                    : `${item.label} (no weights available)`
                  }
                >
                  <img 
                    src={item.imageUrl} 
                    alt={item.label}
                    style={{
                      ...styles.historyImage,
                      ...(isSelected ? styles.historyImageSelected : {})
                    }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'block';
                    }}
                  />
                  <div style={{...styles.historyImagePlaceholder, display: 'none'}}>
                    Image not found
                  </div>
                  <div style={{
                    ...styles.historyLabel,
                    ...(isSelected ? styles.historyLabelSelected : {})
                  }}>
                    {isSelected ? '✓ ' : ''}{item.label}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    padding: '24px',
    margin: '20px 0',
    boxShadow: '0 8px 16px rgba(0,0,0,0.15)',
    color: 'white',
  },
  header: {
    marginBottom: '20px',
  },
  title: {
    margin: '0 0 8px 0',
    fontSize: '20px',
    fontWeight: '700',
    color: 'white',
  },
  description: {
    margin: 0,
    fontSize: '14px',
    color: 'rgba(255,255,255,0.9)',
  },
  workflow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '20px',
    padding: '15px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '8px',
  },
  step: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    borderRadius: '6px',
    background: 'rgba(255,255,255,0.1)',
    fontSize: '13px',
    transition: 'all 0.3s ease',
  },
  stepActive: {
    background: 'rgba(255,255,255,0.3)',
    fontWeight: '600',
  },
  stepComplete: {
    background: 'rgba(76, 175, 80, 0.5)',
  },
  stepNumber: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    background: 'rgba(255,255,255,0.3)',
    fontSize: '12px',
    fontWeight: '700',
  },
  arrow: {
    margin: '0 12px',
    fontSize: '18px',
    opacity: 0.7,
  },
  controls: {
    display: 'flex',
    gap: '12px',
    marginBottom: '16px',
  },
  button: {
    flex: 1,
    padding: '14px 24px',
    fontSize: '15px',
    fontWeight: '600',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: '0 4px 8px rgba(0,0,0,0.15)',
  },
  continueButton: {
    background: 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    color: 'white',
  },
  refineButton: {
    background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    color: 'white',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  status: {
    background: 'rgba(255,255,255,0.25)',
    padding: '12px 16px',
    borderRadius: '6px',
    color: 'white',
    fontSize: '14px',
    marginBottom: '12px',
    fontWeight: '500',
  },
  error: {
    background: 'rgba(244, 67, 54, 0.4)',
    padding: '12px 16px',
    borderRadius: '6px',
    color: 'white',
    fontSize: '14px',
    marginBottom: '12px',
    fontWeight: '500',
  },
  infoBox: {
    background: 'rgba(255,255,255,0.15)',
    borderRadius: '8px',
    padding: '12px 16px',
    fontSize: '13px',
    color: 'rgba(255,255,255,0.95)',
    marginBottom: '12px',
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '6px 0',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
  },
  badge: {
    background: 'rgba(255,255,255,0.3)',
    padding: '2px 10px',
    borderRadius: '12px',
    fontWeight: '700',
    fontSize: '12px',
  },
  helpText: {
    background: 'rgba(255,255,255,0.1)',
    padding: '10px 14px',
    borderRadius: '6px',
    fontSize: '12px',
    lineHeight: '1.5',
    color: 'rgba(255,255,255,0.9)',
  },
  historyContainer: {
    marginTop: '30px',
    padding: '20px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '8px',
  },
  historyTitle: {
    margin: '0 0 8px 0',
    fontSize: '16px',
    fontWeight: '600',
    color: 'white',
  },
  historyDescription: {
    margin: '0 0 16px 0',
    fontSize: '13px',
    color: 'rgba(255,255,255,0.8)',
  },
  historyGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '12px',
  },
  historyItem: {
    position: 'relative',
    aspectRatio: '1',
    borderRadius: '8px',
    overflow: 'hidden',
    cursor: 'pointer',
    border: '2px solid rgba(255,255,255,0.3)',
    transition: 'all 0.2s ease',
    background: 'rgba(0,0,0,0.2)',
  },
  historyItemReference: {
    border: '2px solid rgba(255,215,0,0.6)',
  },
  historyItemDisabled: {
    cursor: 'not-allowed',
    opacity: 0.5,
  },
  historyItemSelected: {
    border: '3px solid #4CAF50',
    boxShadow: '0 0 12px rgba(76, 175, 80, 0.6)',
    transform: 'scale(1.05)',
  },
  historyImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  historyImageSelected: {
    filter: 'brightness(1.1)',
  },
  historyImagePlaceholder: {
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '12px',
    color: 'rgba(255,255,255,0.6)',
    textAlign: 'center',
    padding: '8px',
  },
  historyLabel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: '6px 8px',
    background: 'rgba(0,0,0,0.7)',
    color: 'white',
    fontSize: '12px',
    fontWeight: '600',
    textAlign: 'center',
  },
  historyLabelSelected: {
    background: 'rgba(76, 175, 80, 0.9)',
    color: 'white',
  },
  historyLoading: {
    textAlign: 'center',
    padding: '20px',
    color: 'rgba(255,255,255,0.7)',
    fontSize: '14px',
  },
  historyEmpty: {
    textAlign: 'center',
    padding: '20px',
    color: 'rgba(255,255,255,0.5)',
    fontSize: '14px',
  },
  tagInjectionContainer: {
    marginBottom: '20px',
    padding: '16px',
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '8px',
    border: '1px solid rgba(255,255,255,0.2)',
  },
  tagInjectionTitle: {
    margin: '0 0 6px 0',
    fontSize: '15px',
    fontWeight: '600',
    color: 'white',
  },
  tagInjectionDescription: {
    margin: '0 0 12px 0',
    fontSize: '12px',
    color: 'rgba(255,255,255,0.8)',
  },
  tagInjectionControls: {
    display: 'flex',
    gap: '10px',
    alignItems: 'stretch',
  },
  tagInput: {
    flex: 1,
    padding: '10px 14px',
    fontSize: '14px',
    border: '2px solid rgba(255,255,255,0.3)',
    borderRadius: '6px',
    background: 'rgba(255,255,255,0.95)',
    color: '#333',
    outline: 'none',
    transition: 'all 0.2s ease',
  },
  emphasisSelect: {
    padding: '10px 14px',
    fontSize: '14px',
    border: '2px solid rgba(255,255,255,0.3)',
    borderRadius: '6px',
    background: 'rgba(255,255,255,0.95)',
    color: '#333',
    cursor: 'pointer',
    outline: 'none',
    minWidth: '130px',
    fontWeight: '500',
  },
  tagPreview: {
    marginTop: '10px',
    padding: '8px 12px',
    background: 'rgba(76, 175, 80, 0.3)',
    borderRadius: '6px',
    fontSize: '13px',
    color: 'white',
    border: '1px solid rgba(76, 175, 80, 0.5)',
  },
};

export default RefinementIterationControls;

