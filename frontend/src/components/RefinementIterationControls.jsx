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

import React, { useState } from 'react';

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

  // Check if we can proceed
  const canRefineMore = images && images.length === 4 && selectedImage !== null && !isGenerating;

  const handleRefineMore = async () => {
    if (!canRefineMore) return;

    try {
      setIsGenerating(true);
      setStatus(`🔄 Round ${round + 1}: Learning from your selection...`);
      setError(null);

      // Call unified endpoint: record + propose + generate
      const response = await fetch('/api/pbo/refine-next-round', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          selected_image_id: selectedImage,
          all_image_ids: images.map(img => img.id),
          round_number: round
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to refine: ${response.statusText}`);
      }

      const data = await response.json();
      const newRound = data.round_number;
      setRound(newRound);
      setStatus(`✅ Round ${newRound} complete! PBO generated 4 new variations.`);

      // Notify parent component with new images
      if (onRefinementComplete) {
        const newImages = data.image_paths.map((path, i) => ({
          id: `round_${newRound}_image_${i}`,
          url: path
        }));
        onRefinementComplete(newImages, newRound);
      }

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

      <div style={styles.controls}>
        {/* Continue to Next Stage */}
        <button
          onClick={onContinue}
          disabled={!selectedImage || disabled || isGenerating}
          style={{
            ...styles.button,
            ...styles.continueButton,
            ...((!selectedImage || disabled || isGenerating) ? styles.buttonDisabled : {})
          }}
          title="Proceed to the next stage with this selection"
        >
          Continue to Next Stage →
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
          title="Generate 4 new variations based on your selection"
        >
          {isGenerating ? '⏳ Generating Round ' + (round + 1) + '...' : '🔄 Refine More'}
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
        <div style={styles.infoRow}>
          <span>Round:</span>
          <span style={styles.badge}>{round}</span>
        </div>
        <div style={styles.infoRow}>
          <span>Reference:</span>
          <span>Original {stage} selection (fixed)</span>
        </div>
        <div style={styles.infoRow}>
          <span>Selection:</span>
          <span>{selectedImage ? '✅ Image selected' : '⚠️ Select an image'}</span>
        </div>
      </div>

      {/* Help Text */}
      <div style={styles.helpText}>
        💡 <strong>How it works:</strong> Each image represents different concept weights. 
        Your selection trains PBO to propose better mixtures each round. All rounds use the same reference image with different fused embeddings.
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
};

export default RefinementIterationControls;

