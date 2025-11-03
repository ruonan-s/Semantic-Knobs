import React, { useState, useEffect, useCallback, useRef } from 'react';
import BubbleChart from './BubbleChart';
import ConceptLists from './ConceptLists';
import ImageEffectPreview from './ImageEffectPreview';

/**
 * ConceptRefinementPanel - Integrated panel for preference-driven tag refinement
 * Shows bubble chart, concept lists, and image effect preview
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
  const [concepts, setConcepts] = useState([]);
  const [categorized, setCategorized] = useState({
    positive: [],
    neutral: [],
    negative: []
  });
  const [imageEffects, setImageEffects] = useState({});
  const [incidenceMatrix, setIncidenceMatrix] = useState({});
  const [tagPreferences, setTagPreferences] = useState({});  // NEW: tag_id -> 'positive'|'negative'|null
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPanel, setShowPanel] = useState(false);
  
  // Debounce timer for ranking updates
  const rankingDebounceRef = useRef(null);

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
      
      console.log('[CONCEPT INIT] Response:', {
        success: data.success,
        concept_count: data.concepts?.length || 0,
        categorized: data.categorized,
        sample_concepts: data.concepts?.slice(0, 3)
      });
      
      if (data.success) {
        setConcepts(data.concepts || []);
        setCategorized(data.categorized || { positive: [], neutral: [], negative: [] });
        setImageEffects(data.image_effects || {});
        setIncidenceMatrix(data.incidence_matrix || {});
        const tagPrefs = data.tag_preferences || {};
        setTagPreferences(tagPrefs);
        setIsInitialized(true);
        setShowPanel(data.concepts && data.concepts.length > 0);
        
        // Notify parent of tag preferences
        if (onTagPreferencesUpdate) {
          onTagPreferencesUpdate(tagPrefs);
        }
        
        console.log('[CONCEPT INIT] State updated:', {
          concepts: data.concepts?.length,
          positive: data.categorized?.positive?.length || 0,
          neutral: data.categorized?.neutral?.length || 0,
          negative: data.categorized?.negative?.length || 0,
          tag_preferences: Object.keys(tagPrefs).length
        });
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

    console.log('[TAG INTERACTION] Request:', { tagId, preference });

    try {
      const response = await fetch('/api/concepts/interact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: stage,
          tag_id: tagId,
          preference: preference
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to update concept: ${response.status}`);
      }

      const data = await response.json();
      
      console.log('[TAG INTERACTION] Response:', {
        success: data.success,
        categorized: data.categorized,
        concepts_updated: data.concepts?.length || 0
      });
      
      if (data.success) {
        setConcepts(data.concepts || []);
        setCategorized(data.categorized || { positive: [], neutral: [], negative: [] });
        setImageEffects(data.image_effects || {});
        const tagPrefs = data.tag_preferences || {};
        setTagPreferences(tagPrefs);
        
        // Notify parent of tag preferences
        if (onTagPreferencesUpdate) {
          onTagPreferencesUpdate(tagPrefs);
        }
        
        console.log('[TAG INTERACTION] State updated:', {
          positive: data.categorized?.positive?.length || 0,
          neutral: data.categorized?.neutral?.length || 0,
          negative: data.categorized?.negative?.length || 0,
          tag_preferences: Object.keys(tagPrefs).length
        });
      }
    } catch (err) {
      console.error('Error handling tag interaction:', err);
    }
  }, [sessionId, stage, isInitialized]);

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

    console.log('[IMAGE SELECTION] Image selected:', selectedImage);

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
        
        console.log('[IMAGE SELECTION] Response:', {
          success: data.success,
          categorized: data.categorized
        });
        
        if (data.success) {
          setConcepts(data.concepts || []);
          setCategorized(data.categorized || { positive: [], neutral: [], negative: [] });
          setImageEffects(data.image_effects || {});
          const tagPrefs = data.tag_preferences || {};
          setTagPreferences(tagPrefs);
          
          // Notify parent of tag preferences
          if (onTagPreferencesUpdate) {
            onTagPreferencesUpdate(tagPrefs);
          }
          
          console.log('[IMAGE SELECTION] State updated:', {
            positive: data.categorized?.positive?.length || 0,
            neutral: data.categorized?.neutral?.length || 0,
            negative: data.categorized?.negative?.length || 0,
            tag_preferences: Object.keys(tagPrefs).length
          });
        }
      } catch (err) {
        console.error('Error handling image selection:', err);
      }
    };

    handleImageSelection();
  }, [selectedImage, isInitialized, sessionId, stage]);

  // Handle ranking change (drag and drop)
  const handleRankingChange = useCallback((positiveIds, negativeIds) => {
    console.log('[RANKING CHANGE] Request:', {
      positiveIds,
      negativeIds
    });

    // Debounce the API call
    if (rankingDebounceRef.current) {
      clearTimeout(rankingDebounceRef.current);
    }

    // Optimistically update UI (ONLY positive and negative, preserve neutral)
    setCategorized(prev => {
      console.log('[RANKING CHANGE] Optimistic update:', {
        before: prev,
        after: {
          positive: positiveIds,
          neutral: prev.neutral,  // Preserve neutral!
          negative: negativeIds
        }
      });
      
      return {
        ...prev,
        positive: positiveIds,
        negative: negativeIds
        // Keep neutral unchanged
      };
    });

    rankingDebounceRef.current = setTimeout(async () => {
      try {
        const response = await fetch('/api/concepts/rank', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            stage: stage,
            positive_concept_ids: positiveIds,
            negative_concept_ids: negativeIds
          })
        });

        if (!response.ok) {
          throw new Error(`Failed to update rankings: ${response.status}`);
        }

        const data = await response.json();
        
        console.log('[RANKING CHANGE] Response:', {
          success: data.success,
          categorized: data.categorized
        });
        
        if (data.success) {
          setConcepts(data.concepts || []);
          setCategorized(data.categorized || { positive: [], neutral: [], negative: [] });
          setImageEffects(data.image_effects || {});
          const tagPrefs = data.tag_preferences || {};
          setTagPreferences(tagPrefs);
          
          // Notify parent of tag preferences
          if (onTagPreferencesUpdate) {
            onTagPreferencesUpdate(tagPrefs);
          }
          
          console.log('[RANKING CHANGE] State updated:', {
            positive: data.categorized?.positive?.length || 0,
            neutral: data.categorized?.neutral?.length || 0,
            negative: data.categorized?.negative?.length || 0,
            tag_preferences: Object.keys(tagPrefs).length
          });
        }
      } catch (err) {
        console.error('Error updating rankings:', err);
      }
    }, 200); // 200ms debounce
  }, [sessionId, stage]);

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
            Click tags on images to refine preferences. Drag concepts to reorder importance.
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

      {/* Main Content Area */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '20px',
        marginBottom: '20px'
      }}>
        {/* Left: Bubble Chart */}
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
            concepts={concepts}
            onConceptClick={(bubble) => {
              console.log('Bubble clicked:', bubble);
            }}
          />
        </div>

        {/* Right: Concept Lists */}
        <div style={{ minHeight: '600px' }}>
          <h3 style={{
            margin: '0 0 12px 0',
            fontSize: '16px',
            fontWeight: '600',
            color: '#333'
          }}>
            Concept Categories (Drag to Reorder)
          </h3>
          <ConceptLists
            concepts={concepts}
            categorized={categorized}
            onRankingChange={handleRankingChange}
          />
        </div>
      </div>

      {/* Bottom: Image Effect Preview */}
      <div style={{ marginTop: '20px' }}>
        <ImageEffectPreview
          images={images}
          imageEffects={imageEffects}
          selectedImage={selectedImage}
          onImageClick={onImageSelect}
        />
      </div>
    </div>
  );
}

export default ConceptRefinementPanel;

