/**
 * PBOControls - UI controls for Preferential Bayesian Optimization
 *
 * Provides buttons and status for:
 * - Favorite selection (strong duels)
 * - Generate Next 4 (PBO proposals → SDXL images)
 * - Status display
 */

import React, { useState } from 'react';

const PBOControls = ({
  sessionId,
  stage,
  images,
  selectedImage,
  concepts,
  conceptStates,
  onPBOGenerate,
  disabled = false
}) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState(null);

  // Check if we can select a favorite (need 4 images)
  const canSelectFavorite = images && images.length === 4 && selectedImage !== null;

  // Get current weights from concept states
  const getCurrentWeights = () => {
    if (!concepts || !conceptStates) return null;

    return concepts.map(concept => {
      const state = conceptStates[concept.id];
      return state ? state.ema_w : (1.0 / concepts.length);
    });
  };

  // Get negative concept IDs (where dislike > like)
  const getNegativeConceptIds = () => {
    if (!concepts || !conceptStates) return [];

    return concepts
      .filter(concept => {
        const state = conceptStates[concept.id];
        return state && state.dislike_count > state.like_count;
      })
      .map(concept => concept.id);
  };

  const handleSelectFavorite = async () => {
    if (!canSelectFavorite) return;

    try {
      setStatus('Recording favorite selection...');
      setError(null);

      const response = await fetch('/api/pbo/favorite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          favorite_image_id: selectedImage,
          all_image_ids: images.map(img => img.id)
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to record favorite: ${response.statusText}`);
      }

      const data = await response.json();
      setStatus(`✅ Favorite recorded: ${data.num_duels} preferences added`);

      // Auto-clear status after 3s
      setTimeout(() => setStatus(''), 3000);

    } catch (err) {
      console.error('Error recording favorite:', err);
      setError(err.message);
      setStatus('');
    }
  };

  const handleGenerateNext4 = async () => {
    try {
      setIsGenerating(true);
      setStatus('Generating proposals with PBO...');
      setError(null);

      const w_current = getCurrentWeights();
      const negatives = getNegativeConceptIds();

      // Step 1: Propose
      const proposeResponse = await fetch('/api/pbo/propose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          w_current: w_current,
          negatives: negatives
        })
      });

      if (!proposeResponse.ok) {
        throw new Error(`Failed to propose: ${proposeResponse.statusText}`);
      }

      const proposeData = await proposeResponse.json();
      setStatus(`✅ Generated ${proposeData.proposals.length} proposals. Generating images...`);

      // Step 2: Generate images
      const generateResponse = await fetch('/api/pbo/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          proposals: proposeData.proposals,
          seed_base: Math.floor(Math.random() * 10000)
        })
      });

      if (!generateResponse.ok) {
        throw new Error(`Failed to generate: ${generateResponse.statusText}`);
      }

      const generateData = await generateResponse.json();
      setStatus(`✅ Generated ${generateData.image_paths.length} images!`);

      // Notify parent component
      if (onPBOGenerate) {
        onPBOGenerate(generateData);
      }

      // Auto-clear status after 5s
      setTimeout(() => setStatus(''), 5000);

    } catch (err) {
      console.error('Error generating with PBO:', err);
      setError(err.message);
      setStatus('');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>🎯 Preferential Bayesian Optimization</h3>
        <p style={styles.description}>
          Use your preferences to intelligently explore design variations
        </p>
      </div>

      <div style={styles.controls}>
        {/* Favorite Button */}
        <button
          onClick={handleSelectFavorite}
          disabled={!canSelectFavorite || disabled || isGenerating}
          style={{
            ...styles.button,
            ...styles.favoriteButton,
            ...((!canSelectFavorite || disabled || isGenerating) ? styles.buttonDisabled : {})
          }}
          title={!canSelectFavorite ? "Select one of the 4 images first" : "Mark as favorite"}
        >
          ⭐ Mark as Favorite
        </button>

        {/* Generate Next 4 Button */}
        <button
          onClick={handleGenerateNext4}
          disabled={disabled || isGenerating || !concepts || concepts.length === 0}
          style={{
            ...styles.button,
            ...styles.generateButton,
            ...((disabled || isGenerating || !concepts) ? styles.buttonDisabled : {})
          }}
          title="Generate 4 new variants using PBO"
        >
          {isGenerating ? '⏳ Generating...' : '🚀 Generate Next 4'}
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
          <span>Current Round:</span>
          <span>{images ? Math.ceil(images.length / 4) : 0}</span>
        </div>
        <div style={styles.infoRow}>
          <span>Negative Concepts:</span>
          <span>{getNegativeConceptIds().length}</span>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    borderRadius: '12px',
    padding: '20px',
    margin: '20px 0',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
  },
  header: {
    marginBottom: '15px',
  },
  title: {
    margin: '0 0 8px 0',
    fontSize: '18px',
    fontWeight: '600',
    color: 'white',
  },
  description: {
    margin: 0,
    fontSize: '13px',
    color: 'rgba(255,255,255,0.9)',
  },
  controls: {
    display: 'flex',
    gap: '10px',
    marginBottom: '15px',
  },
  button: {
    flex: 1,
    padding: '12px 20px',
    fontSize: '14px',
    fontWeight: '600',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  },
  favoriteButton: {
    background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    color: 'white',
  },
  generateButton: {
    background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    color: 'white',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  status: {
    background: 'rgba(255,255,255,0.2)',
    padding: '10px 15px',
    borderRadius: '6px',
    color: 'white',
    fontSize: '13px',
    marginBottom: '10px',
  },
  error: {
    background: 'rgba(255,100,100,0.3)',
    padding: '10px 15px',
    borderRadius: '6px',
    color: 'white',
    fontSize: '13px',
    marginBottom: '10px',
  },
  infoBox: {
    background: 'rgba(255,255,255,0.1)',
    borderRadius: '6px',
    padding: '10px 15px',
    fontSize: '12px',
    color: 'rgba(255,255,255,0.9)',
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '4px 0',
  },
};

export default PBOControls;
