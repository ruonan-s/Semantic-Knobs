import React, { useState, useEffect, useCallback, useRef } from 'react';
import BubbleChart from './BubbleChart';

/**
 * ConceptRefinementPanel - Integrated panel for preference-driven tag refinement
 * Shows bubble chart for concept weight visualization
 */
function ConceptRefinementPanel({ 
  sessionId, 
  stage, 
  images, 
  selectedImage,
  onImageSelect,
  onTagClick,  // Called when a tag is clicked, triggers concept update
  onTagPreferencesUpdate  // NEW: Called when tag preferences change
}) {
  const prevSelectedImageRef = useRef(null);
  const abortControllerRef = useRef(null);  // For cancelling in-flight requests
  const [concepts, setConcepts] = useState([]);
  const [tagPreferences, setTagPreferences] = useState({});  // tag_id -> 'positive'|'negative'|null
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPanel, setShowPanel] = useState(false);
  const [conceptsUpdateKey, setConceptsUpdateKey] = useState(0);  // Force re-render key

  // Monitor concept changes for debugging (minimal logging for performance)
  useEffect(() => {
    if (concepts.length > 0) {
      console.log(`[CONCEPTS] Updated: ${concepts.length} concepts, key=${conceptsUpdateKey}`);
    }
  }, [concepts, conceptsUpdateKey]);

  // Cleanup: Cancel any pending requests on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        console.log('[CLEANUP] Cancelled pending requests on unmount');
      }
    };
  }, []);

  // Initialize concepts when component mounts or stage changes
  useEffect(() => {
    if (sessionId && stage && images && images.length > 0) {
      initializeConcepts();
    }
  }, [sessionId, stage, images]);

  const initializeConcepts = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const imageIds = images.map(img => img.id);
      
      const response = await fetch('/api/concepts/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          image_ids: imageIds
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to initialize concepts: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setConcepts(data.concepts || []);
        setConceptsUpdateKey(prev => prev + 1);  // Force BubbleChart re-render
        
        const tagPrefs = data.tag_preferences || {};
        setTagPreferences(tagPrefs);
        setIsInitialized(true);
        setShowPanel(data.concepts && data.concepts.length > 0);
        
        // Notify parent of tag preferences
        if (onTagPreferencesUpdate) {
          onTagPreferencesUpdate(tagPrefs);
        }
        
        console.log(`[CONCEPT INIT] Initialized ${data.concepts?.length || 0} concepts`);
      } else {
        console.warn('Concept initialization returned success=false');
        setShowPanel(false);
      }
    } catch (err) {
      console.error('Error initializing concepts:', err);
      setError(err.message);
      setShowPanel(false);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle tag interaction (like/dislike)
  const handleTagInteraction = useCallback(async (tagId, preference) => {
    if (!isInitialized) return;

    // Cancel previous request if it exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      console.log('[TAG INTERACTION] ⏹️ Cancelled previous request');
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();

    // DIRECT SERVER UPDATE: Backend is fast enough (~50-100ms), no optimistic update needed
    try {
      const response = await fetch('/api/concepts/interact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          tag_id: tagId,
          preference: preference
        }),
        signal: abortControllerRef.current.signal  // Allow cancellation
      });

      if (!response.ok) {
        throw new Error(`Failed to update concept: ${response.status}`);
      }

      const data = await response.json();
      
      // Update state with server response (single source of truth)
      if (data.success) {
        setConcepts(data.concepts || []);
        setConceptsUpdateKey(prev => prev + 1);  // Force BubbleChart re-render
        
        const tagPrefs = data.tag_preferences || {};
        setTagPreferences(tagPrefs);
        
        // Notify parent with server data
        if (onTagPreferencesUpdate) {
          onTagPreferencesUpdate(tagPrefs);
        }
        
        console.log('[TAG INTERACTION] ✅ Updated:', data.concepts?.length, 'concepts');
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // Request was cancelled, this is expected
        console.log('[TAG INTERACTION] ⏹️ Request aborted (newer request in progress)');
      } else {
        console.error('[TAG INTERACTION] Error:', err);
      }
    } finally {
      // Clear the abort controller reference
      if (abortControllerRef.current) {
        abortControllerRef.current = null;
      }
    }
  }, [sessionId, stage, isInitialized, onTagPreferencesUpdate]);

  // Expose tag interaction handler to parent via callback
  useEffect(() => {
    if (onTagClick && isInitialized) {
      // Register this handler so App.jsx can call it
      onTagClick.current = handleTagInteraction;
    }
  }, [handleTagInteraction, onTagClick, isInitialized]);

  // Handle image selection changes
  useEffect(() => {
    if (!isInitialized || !selectedImage) return;
    
    // Only trigger if selection actually changed
    if (prevSelectedImageRef.current === selectedImage) return;
    
    prevSelectedImageRef.current = selectedImage;

    const handleImageSelection = async () => {
      try {
        const response = await fetch('/api/concepts/select-image', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            stage: stage,
            image_id: selectedImage,
            boost_amount: 0.5
          })
        });

        if (!response.ok) {
          throw new Error(`Failed to handle image selection: ${response.status}`);
        }

        const data = await response.json();
        
        if (data.success) {
          setConcepts(data.concepts || []);
          setConceptsUpdateKey(prev => prev + 1);  // Force BubbleChart re-render
          
          const tagPrefs = data.tag_preferences || {};
          setTagPreferences(tagPrefs);
          
          // Notify parent of tag preferences
          if (onTagPreferencesUpdate) {
            onTagPreferencesUpdate(tagPrefs);
          }
        }
      } catch (err) {
        console.error('Error handling image selection:', err);
      }
    };

    handleImageSelection();
  }, [selectedImage, isInitialized, sessionId, stage]);

  if (!showPanel) {
    return null;
  }

  if (isLoading && !isInitialized) {
    return (
      <div style={{
        padding: '40px',
        textAlign: 'center',
        backgroundColor: '#f9f9f9',
        borderRadius: '8px',
        margin: '20px 0'
      }}>
        <div style={{ fontSize: '16px', color: '#666' }}>
          Building concept map from image tags...
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        padding: '20px',
        backgroundColor: '#ffebee',
        borderRadius: '8px',
        margin: '20px 0',
        border: '1px solid #ffcdd2'
      }}>
        <div style={{ fontSize: '14px', color: '#c62828' }}>
          Error: {error}
        </div>
      </div>
    );
  }

  if (!isInitialized || concepts.length === 0) {
    return null;
  }

  return (
    <div style={{
      marginTop: '30px',
      padding: '20px',
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      border: '2px solid #e0e0e0',
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        paddingBottom: '15px',
        borderBottom: '2px solid #e0e0e0'
      }}>
        <div>
          <h2 style={{
            margin: '0 0 8px 0',
            fontSize: '20px',
            fontWeight: '600',
            color: '#333'
          }}>
            Preference Refinement
          </h2>
          <p style={{
            margin: 0,
            fontSize: '13px',
            color: '#666'
          }}>
            Click tags on images to refine preferences. Bubble size represents concept weight.
          </p>
        </div>
        <div style={{
          padding: '8px 16px',
          backgroundColor: '#e3f2fd',
          borderRadius: '6px',
          fontSize: '13px',
          color: '#1976d2'
        }}>
          {concepts.length} concepts identified
        </div>
      </div>

      {/* Bubble Chart */}
      <div style={{ minHeight: '600px' }}>
        <h3 style={{
          margin: '0 0 12px 0',
          fontSize: '16px',
          fontWeight: '600',
          color: '#333'
        }}>
          Concept Weight Visualization
        </h3>
        <BubbleChart 
          key={`bubble-chart-${conceptsUpdateKey}`}
          concepts={concepts}
          onConceptClick={(bubble) => {
            console.log('Bubble clicked:', bubble);
          }}
        />
      </div>
    </div>
  );
}

export default ConceptRefinementPanel;

