import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ProgressBar from './components/ProgressBar';
import GenerationStatus from './components/GenerationStatus';
import TagSidebar from './components/TagSidebar';
import JsonPanel from './components/JsonPanel';
import InlineTagDisplay from './components/InlineTagDisplay';
import ConceptRefinementPanel from './components/ConceptRefinementPanel';

/**
 * Evaluation Prototype App
 * 
 * Simplified flow that skips refinement:
 * 1. Landing - Select predefined session
 * 2. Exploration - Same as FULL implementation with bubble chart
 * 3. Slider Generation - Uses exploration weights directly (no refinement)
 */
function App() {
  // Session state
  const [sessionId, setSessionId] = useState(null);
  const [stage, setStage] = useState('landing');
  const [descriptor, setDescriptor] = useState('');
  const [adjective, setAdjective] = useState('');
  const [location, setLocation] = useState('');
  
  // Exploration state
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageTagsMap, setImageTagsMap] = useState({});
  const [showTagsByDefault, setShowTagsByDefault] = useState(true);
  const [conceptSystemReady, setConceptSystemReady] = useState(false);
  const [conceptTagPreferences, setConceptTagPreferences] = useState({});
  
  // UI state
  const [statusMessages, setStatusMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [buttonColor, setButtonColor] = useState('#007bff');
  const [predefinedSessions, setPredefinedSessions] = useState([]);
  const [userPreferences, setUserPreferences] = useState({ selections: {}, tags: {} });
  
  // Drawer/Panel state
  const [showTagDrawer, setShowTagDrawer] = useState(false);
  const [showJsonPanel, setShowJsonPanel] = useState(false);
  const [currentImageTags, setCurrentImageTags] = useState([]);
  const [currentImageId, setCurrentImageId] = useState(null);
  const [currentImageJson, setCurrentImageJson] = useState(null);
  const [drawerPosition, setDrawerPosition] = useState({ top: 0, bottom: 0, left: 0 });
  
  // Slider state
  const [sliderRows, setSliderRows] = useState([]);
  const [sliderNewLocation, setSliderNewLocation] = useState('');
  const [isGeneratingSlider, setIsGeneratingSlider] = useState(false);
  
  // User ID for evaluation tracking
  const [userId, setUserId] = useState('');
  
  const imageRefs = useRef({});
  const conceptTagHandlerRef = useRef(null);

  // Add global styles
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      * {
        font-family: SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif !important;
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // Reset concept system when stage changes
  useEffect(() => {
    console.log('[APP] Stage changed to:', stage, '- Resetting concept system ready flag');
    setConceptSystemReady(false);
    conceptTagHandlerRef.current = null;
  }, [stage]);

  // Load predefined sessions on mount
  useEffect(() => {
    if (stage === 'landing') {
      fetchPredefinedSessions();
    }
  }, [stage]);

  // Load tags when images change
  useEffect(() => {
    if (images && images.length > 0 && sessionId && showTagsByDefault) {
      loadAllImageTags(images);
    }
  }, [images, sessionId, showTagsByDefault]);

  const addStatusMessage = (message) => {
    setStatusMessages(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
  };

  const fetchPredefinedSessions = async () => {
    try {
      const res = await fetch('/api/eval/predefined-sessions');
      if (res.ok) {
        const data = await res.json();
        setPredefinedSessions(data.sessions || []);
      }
    } catch (error) {
      console.error('Failed to fetch predefined sessions:', error);
      addStatusMessage(`Error loading sessions: ${error.message}`);
    }
  };

  const loadPredefinedSession = async (sessionName) => {
    setIsLoading(true);
    try {
      addStatusMessage(`Loading session: ${sessionName}...`);
      
      const res = await fetch('/api/eval/load-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_name: sessionName,
          user_id: userId || null
        })
      });

      if (!res.ok) {
        throw new Error(`Failed to load session: ${res.status}`);
      }

      const data = await res.json();
      
      setSessionId(data.session_id);
      setDescriptor(data.descriptor);
      setAdjective(data.adjective);
      setLocation(data.location);
      setImages(data.images);
      setStage('impression');
      
      addStatusMessage(`Session loaded: ${data.descriptor}`);
      addStatusMessage(`${data.images.length} images ready for exploration`);
      
    } catch (error) {
      console.error('Error loading session:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const loadAllImageTags = async (imagesList) => {
    if (!imagesList || imagesList.length === 0) return;
    
    try {
      const tagPromises = imagesList.map(async (image) => {
        try {
          const res = await fetch('/api/tags', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              stage: 'impression',
              image_id: image.id
            })
          });
          
          if (!res.ok) throw new Error(`Failed to load tags for ${image.id}`);
          
          const data = await res.json();
          return { imageId: image.id, tags: data.tags };
        } catch (error) {
          console.error(`Failed to load tags for ${image.id}:`, error);
          return { imageId: image.id, tags: [] };
        }
      });

      const tagResults = await Promise.all(tagPromises);
      const newTagsMap = {};
      tagResults.forEach(({ imageId, tags }) => {
        newTagsMap[imageId] = tags;
      });
      
      setImageTagsMap(prev => ({ ...prev, ...newTagsMap }));
    } catch (error) {
      console.error('Failed to load some image tags:', error);
    }
  };

  // Load tags for drawer
  const loadImageTags = async (imageId, event) => {
    try {
      const imageElement = imageRefs.current[imageId];
      if (imageElement) {
        const rect = imageElement.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        setDrawerPosition({
          top: rect.top + scrollTop,
          bottom: rect.bottom + scrollTop,
          left: rect.left
        });
      }

      const res = await fetch('/api/tags', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: 'impression',
          image_id: imageId
        })
      });
      
      if (!res.ok) throw new Error('Failed to load tags');
      
      const data = await res.json();
      setCurrentImageTags(data.tags);
      setCurrentImageId(imageId);
      setShowTagDrawer(true);
    } catch (error) {
      console.error('Failed to load image tags:', error);
      addStatusMessage(`Failed to load tags: ${error.message}`);
    }
  };

  // Load JSON for image
  const loadImageJson = async (imageId) => {
    try {
      const res = await fetch('/api/json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          stage: 'impression',
          image_id: imageId
        })
      });
      
      if (!res.ok) throw new Error('Failed to load JSON');
      
      const data = await res.json();
      setCurrentImageJson(data.json_data);
      setCurrentImageId(imageId);
      setShowJsonPanel(true);
    } catch (error) {
      console.error('Failed to load JSON:', error);
      addStatusMessage(`Failed to load JSON: ${error.message}`);
    }
  };

  const handleSelect = (imageId) => {
    setSelectedImage(imageId);
    addStatusMessage(`Selected: ${imageId}`);
  };

  // Helper function to normalize tags
  const normalizeTag = (tag) => {
    if (typeof tag !== 'string') return '';
    return tag.trim().toLowerCase();
  };

  // Callback for concept system preferences update
  const handleConceptTagPreferencesUpdate = useCallback((tagPrefs) => {
    console.log('[APP] Concept tag preferences updated:', Object.keys(tagPrefs).length);
    setConceptTagPreferences({ ...tagPrefs });
    
    if (!conceptSystemReady) {
      console.log('[APP] Concept system is now ready');
      setConceptSystemReady(true);
    }
  }, [conceptSystemReady]);

  // Handle tag preference (concept-based)
  const handleTagPreference = useCallback((tag, preference, imageId) => {
    console.log('[TAG CLICK]', { tag, preference, imageId });
    
    const imageTags = imageTagsMap[imageId] || [];
    let tagIndex = imageTags.findIndex(t => t === tag);
    
    if (tagIndex === -1) {
      const normalizedTag = normalizeTag(tag);
      tagIndex = imageTags.findIndex(t => normalizeTag(t) === normalizedTag);
    }
    
    if (tagIndex === -1) {
      addStatusMessage(`Unable to set preference for "${tag}". Tag not found.`);
      return;
    }
    
    const tagId = `tag_impression_${imageId}_${tagIndex}`;
    
    if (!conceptSystemReady || !conceptTagHandlerRef.current) {
      addStatusMessage('Concept system is initializing... Please wait a moment.');
      return;
    }
    
    conceptTagHandlerRef.current(tagId, preference);
    
    const prefEmoji = preference === 'positive' ? '👍' : '👎';
    addStatusMessage(`${prefEmoji} Set "${tag}" as ${preference}`);
  }, [imageTagsMap, conceptSystemReady]);

  // Derive UI preferences from concept system
  const derivedTagPreferences = useMemo(() => {
    const derivedPrefs = {
      tags: {},
      currentStage: 'impression'
    };

    for (const [tagId, preference] of Object.entries(conceptTagPreferences)) {
      if (!preference) continue;
      
      const parts = tagId.split('_');
      if (parts.length < 4 || parts[0] !== 'tag') continue;
      
      const tagStage = parts[1];
      const tagIndex = parseInt(parts[parts.length - 1]);
      const imageId = parts.slice(2, parts.length - 1).join('_');
      
      const imageTags = imageTagsMap[imageId] || [];
      const tagText = imageTags[tagIndex];
      
      if (!tagText) continue;
      
      if (!derivedPrefs.tags[tagStage]) {
        derivedPrefs.tags[tagStage] = [];
      }
      
      derivedPrefs.tags[tagStage].push({
        tag: tagText,
        preference: preference,
        source_image: imageId
      });
    }

    return derivedPrefs;
  }, [conceptTagPreferences, imageTagsMap]);

  const handleSkipToSlider = async () => {
    if (!selectedImage) {
      addStatusMessage('Please select an image before continuing');
      return;
    }

    setIsLoading(true);
    try {
      addStatusMessage('Skipping refinement, preparing slider generation...');
      
      const res = await fetch('/api/eval/skip-to-slider', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          selected_image_id: selectedImage
        })
      });

      if (!res.ok) {
        throw new Error(`Failed to skip to slider: ${res.status}`);
      }

      const data = await res.json();
      
      if (data.success) {
        addStatusMessage(data.message);
        setStage('slider_generation');
        setSliderRows([]);
      }
      
    } catch (error) {
      console.error('Error skipping to slider:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const generateSlider = async (newLocation = '') => {
    if (!sessionId) return;
    
    setIsGeneratingSlider(true);
    try {
      addStatusMessage(`Generating slider for ${newLocation || location}...`);
      
      const res = await fetch('/api/generate-slider', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          location: newLocation
        })
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.success && data.sliders) {
        const newRows = data.sliders.map(slider => ({
          slider_type: slider.slider_type,
          adjective: slider.adjective,
          location: slider.location,
          descriptor: slider.descriptor,
          images: slider.images
        }));
        
        setSliderRows(prev => [...prev, ...newRows]);
        addStatusMessage(`Generated ${data.sliders.length} sliders`);
        setSliderNewLocation('');
      }
    } catch (error) {
      console.error('Slider generation error:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsGeneratingSlider(false);
    }
  };

  // Custom stages for eval - only landing, impression, slider_generation
  const evalStages = ['landing', 'impression', 'slider_generation'];

  // ============== RENDER ==============

  return (
    <div style={{ 
      padding: '20px',
      fontFamily: 'SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif'
    }}>
      
      {/* Progress bar */}
      <ProgressBar currentStage={stage} />
      
      {stage === 'landing' ? (
        // ============== LANDING PAGE ==============
        <div style={{ textAlign: 'center', maxWidth: '800px', margin: '0 auto' }}>
          <h1 style={{ marginBottom: '30px', color: '#333' }}>Evaluation Prototype</h1>
          <p style={{ marginBottom: '30px', color: '#666', fontSize: '18px' }}>
            Explore predefined sessions and generate semantic sliders (no refinement)
          </p>

          {/* User ID input */}
          <div style={{
            marginBottom: '30px',
            padding: '20px',
            backgroundColor: '#f8f9fa',
            borderRadius: '12px',
            border: '1px solid #dee2e6'
          }}>
            <label style={{ display: 'block', marginBottom: '10px', fontWeight: '500', color: '#495057' }}>
              Enter your User ID (optional, for logging):
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., participant_01"
              style={{
                width: '300px',
                padding: '12px 16px',
                fontSize: '16px',
                borderRadius: '8px',
                border: '1px solid #ced4da',
                outline: 'none'
              }}
            />
          </div>

          {/* Predefined sessions grid */}
          <div style={{
            padding: '30px',
            backgroundColor: '#e8f4f8',
            borderRadius: '12px',
            border: '2px solid #007bff'
          }}>
            <h3 style={{ margin: '0 0 20px 0', color: '#333', fontSize: '18px' }}>
              📂 Select a Predefined Session
            </h3>

            {predefinedSessions.length === 0 ? (
              <p style={{ color: '#666' }}>
                No predefined sessions found. Add sessions to eval/predefined_input/
              </p>
            ) : (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
                gap: '15px'
              }}>
                {predefinedSessions.map((session) => (
                  <button
                    key={session.name}
                    onClick={() => loadPredefinedSession(session.name)}
                    disabled={isLoading || !session.valid}
                    style={{
                      padding: '20px',
                      backgroundColor: session.valid ? '#fff' : '#f8f8f8',
                      border: session.valid ? '2px solid #28a745' : '2px solid #dc3545',
                      borderRadius: '12px',
                      cursor: session.valid ? 'pointer' : 'not-allowed',
                      opacity: session.valid ? 1 : 0.6,
                      transition: 'all 0.2s ease',
                      textAlign: 'left'
                    }}
                    onMouseEnter={(e) => {
                      if (session.valid) {
                        e.currentTarget.style.backgroundColor = '#e8f5e9';
                        e.currentTarget.style.transform = 'translateY(-2px)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (session.valid) {
                        e.currentTarget.style.backgroundColor = '#fff';
                        e.currentTarget.style.transform = 'translateY(0)';
                      }
                    }}
                  >
                    <div style={{ fontSize: '16px', fontWeight: '600', color: '#333', marginBottom: '6px' }}>
                      {session.descriptor || session.name}
                    </div>
                    <div style={{ fontSize: '13px', color: '#666' }}>
                      {session.adjective && session.location ? 
                        `${session.adjective} ${session.location}` : 
                        session.name}
                    </div>
                    {!session.valid && (
                      <div style={{ fontSize: '11px', color: '#dc3545', marginTop: '6px' }}>
                        Missing required files
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : stage === 'slider_generation' ? (
        // ============== SLIDER GENERATION ==============
        <div>
          <h2>Semantic Slider Generation</h2>
          <p style={{ color: '#666', marginBottom: '20px' }}>
            Using exploration weights (no refinement) | Session: {sessionId}
          </p>
          
          {/* Loading indicator */}
          {isGeneratingSlider && (
            <div style={{
              padding: '20px',
              textAlign: 'center',
              backgroundColor: '#e7f3ff',
              borderRadius: '8px',
              marginBottom: '20px'
            }}>
              <div style={{ fontSize: '18px', color: '#004085', marginBottom: '10px' }}>
                Generating slider images...
              </div>
              <div style={{ 
                width: '50px', 
                height: '50px', 
                border: '4px solid #f3f3f3',
                borderTop: '4px solid #007bff',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite',
                margin: '0 auto'
              }} />
              <style>{`
                @keyframes spin {
                  0% { transform: rotate(0deg); }
                  100% { transform: rotate(360deg); }
                }
              `}</style>
            </div>
          )}
          
          {/* Generate button for original location */}
          {sliderRows.length === 0 && !isGeneratingSlider && (
            <div style={{
              padding: '30px',
              backgroundColor: '#f8f9fa',
              borderRadius: '12px',
              textAlign: 'center',
              marginBottom: '20px'
            }}>
              <p style={{ marginBottom: '15px', fontSize: '16px', color: '#333' }}>
                Generate semantic slider for: <strong>{adjective} {location}</strong>
              </p>
              <button
                onClick={() => generateSlider('')}
                style={{
                  padding: '14px 32px',
                  fontSize: '18px',
                  backgroundColor: '#007bff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer'
                }}
              >
                Generate Slider
              </button>
            </div>
          )}
          
          {/* Slider display - single slider with exploration weights */}
          {sliderRows.map((slider, rowIndex) => {
            const numImages = slider.images.length;
            
            return (
              <div key={rowIndex} style={{
                marginBottom: '30px',
                padding: '20px',
                backgroundColor: '#f8f9fa',
                borderRadius: '12px',
                border: '1px solid #dee2e6'
              }}>
                {/* Slider header */}
                <div style={{ marginBottom: '15px' }}>
                  <div style={{ fontSize: '16px', fontWeight: '600', color: '#333', marginBottom: '5px' }}>
                    Semantic Slider: {slider.descriptor}
                  </div>
                  <div style={{ fontSize: '13px', color: '#666' }}>
                    Using exploration weights (no refinement)
                  </div>
                </div>
                
                {/* Alpha scale header */}
                <div style={{ marginBottom: '15px' }}>
                  {/* Alpha labels */}
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: '4px',
                    padding: '0 120px 0 10px'
                  }}>
                    {numImages === 6 ? (
                      <>
                        <span style={{ fontSize: '13px', color: '#666' }}>α = 0</span>
                        <span style={{ fontSize: '13px', color: '#666' }}>0.25</span>
                        <span style={{ fontSize: '13px', color: '#666' }}>0.50</span>
                        <span style={{ fontSize: '13px', color: '#666' }}>0.75</span>
                        <span style={{ fontSize: '13px', color: '#666' }}>α = 1</span>
                        <span style={{ fontSize: '13px', color: '#666' }}>α = 1 (ref)</span>
                      </>
                    ) : (
                      slider.images.map((img, idx) => (
                        <span key={idx} style={{ fontSize: '13px', color: '#666' }}>
                          {idx === 0 ? 'α = 0' : idx === numImages - 1 ? 'α = 1' : img.alpha.toFixed(2)}
                        </span>
                      ))
                    )}
                  </div>
                  
                  {/* Arrow line with dots */}
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0 120px 0 10px',
                    marginBottom: '5px'
                  }}>
                    <span style={{ fontWeight: 'bold', color: '#333', fontSize: '14px' }}>
                      generic
                    </span>
                    <div style={{ 
                      flex: 1, 
                      display: 'flex', 
                      alignItems: 'center',
                      margin: '0 15px',
                      position: 'relative'
                    }}>
                      {/* Left arrow */}
                      <div style={{
                        width: 0,
                        height: 0,
                        borderTop: '5px solid transparent',
                        borderBottom: '5px solid transparent',
                        borderRight: '8px solid #666'
                      }} />
                      {/* Line */}
                      <div style={{ 
                        flex: 1, 
                        height: '2px', 
                        backgroundColor: '#666',
                        position: 'relative'
                      }}>
                        {/* Dots */}
                        {[0.2, 0.4, 0.6, 0.8].map((pos, i) => (
                          <div key={i} style={{
                            position: 'absolute',
                            left: `${pos * 100}%`,
                            top: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '8px',
                            height: '8px',
                            backgroundColor: '#666',
                            borderRadius: '50%'
                          }} />
                        ))}
                      </div>
                      {/* Right arrow */}
                      <div style={{
                        width: 0,
                        height: 0,
                        borderTop: '5px solid transparent',
                        borderBottom: '5px solid transparent',
                        borderLeft: '8px solid #666'
                      }} />
                    </div>
                    <span style={{ fontWeight: 'bold', color: '#333', fontSize: '14px' }}>
                      personalized
                    </span>
                  </div>
                </div>
                
                {/* Images row with label */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {/* Images */}
                  <div style={{ display: 'flex', gap: '10px', flex: 1 }}>
                    {slider.images.map((img, imgIndex) => (
                      <div key={imgIndex} style={{ 
                        flex: 1,
                        aspectRatio: '1',
                        borderRadius: '8px',
                        overflow: 'hidden',
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                        border: imgIndex === numImages - 1 && numImages === 6 ? '2px solid #28a745' : 'none'
                      }}>
                        <img 
                          src={img.url} 
                          alt={`Alpha ${img.alpha}${imgIndex === numImages - 1 && numImages === 6 ? ' (reference)' : ''}`}
                          style={{ 
                            width: '100%', 
                            height: '100%', 
                            objectFit: 'cover' 
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  
                  {/* Label on the right */}
                  <div style={{ 
                    width: '120px',
                    textAlign: 'center',
                    padding: '10px'
                  }}>
                    <div style={{ 
                      fontSize: '18px', 
                      fontWeight: 'bold', 
                      color: '#333',
                      textTransform: 'capitalize'
                    }}>
                      {slider.adjective}
                    </div>
                    <div style={{ 
                      fontSize: '16px', 
                      color: '#666',
                      textTransform: 'capitalize'
                    }}>
                      {slider.location}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
          
          {/* New location input */}
          {sliderRows.length > 0 && (
            <div style={{
              display: 'flex',
              gap: '10px',
              alignItems: 'center',
              padding: '20px',
              backgroundColor: '#fff',
              borderRadius: '8px',
              border: '1px solid #dee2e6'
            }}>
              <label style={{ fontWeight: '500', color: '#374151', whiteSpace: 'nowrap' }}>
                New Location:
              </label>
              <input
                type="text"
                value={sliderNewLocation}
                onChange={(e) => setSliderNewLocation(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && sliderNewLocation.trim()) {
                    generateSlider(sliderNewLocation.trim());
                  }
                }}
                placeholder="e.g., bedroom, kitchen, cafe..."
                style={{
                  flex: 1,
                  padding: '10px',
                  fontSize: '16px',
                  borderRadius: '4px',
                  border: '1px solid #ccc'
                }}
                disabled={isGeneratingSlider}
              />
              <button
                onClick={() => generateSlider(sliderNewLocation.trim())}
                disabled={!sliderNewLocation.trim() || isGeneratingSlider}
                style={{
                  padding: '10px 20px',
                  fontSize: '16px',
                  backgroundColor: (!sliderNewLocation.trim() || isGeneratingSlider) ? '#ccc' : '#007bff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: (!sliderNewLocation.trim() || isGeneratingSlider) ? 'not-allowed' : 'pointer'
                }}
              >
                {isGeneratingSlider ? 'Generating...' : 'Generate'}
              </button>
            </div>
          )}
          
          {/* Back button */}
          <button
            onClick={() => setStage('landing')}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer'
            }}
          >
            ← Start New Session
          </button>
        </div>
      ) : (
        // ============== EXPLORATION STAGE (IMPRESSION) ==============
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2>Exploration Stage: Impression</h2>
            <div style={{ color: '#666', fontSize: '14px' }}>
              {descriptor} | Session: {sessionId}
            </div>
          </div>

          {/* Toggle button for tags */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '20px',
            padding: '10px',
            backgroundColor: '#f8f9fa',
            borderRadius: '8px',
            border: '1px solid #dee2e6'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontWeight: '500', color: '#495057' }}>
                Tags Display: {showTagsByDefault ? 'Expanded' : 'Collapsed'}
              </span>
              <span style={{
                fontSize: '12px',
                padding: '3px 8px',
                borderRadius: '12px',
                backgroundColor: conceptSystemReady ? '#d4edda' : '#fff3cd',
                color: conceptSystemReady ? '#155724' : '#856404',
                border: `1px solid ${conceptSystemReady ? '#c3e6cb' : '#ffeeba'}`,
                fontWeight: '500'
              }}>
                {conceptSystemReady ? '✓ Ready' : '⏳ Initializing...'}
              </span>
            </div>
            <button
              onClick={() => setShowTagsByDefault(!showTagsByDefault)}
              style={{
                background: showTagsByDefault ? '#dc3545' : '#28a745',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                padding: '8px 16px',
                fontSize: '14px',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              {showTagsByDefault ? 'Collapse Tags' : 'Expand Tags'}
            </button>
          </div>

          {/* Two-column layout: Images on left, Bubble chart on right */}
          <div style={{ display: 'flex', gap: '20px', marginBottom: '20px' }}>
            {/* Left column: 2x2 grid of images with tags */}
            <div style={{ flex: '0 0 50%', minWidth: '0' }}>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(2, 1fr)', 
                gridTemplateRows: 'repeat(2, 1fr)',
                gap: '15px',
                height: '100%'
              }}>
                {images.map((image) => (
                  <div
                    key={image.id}
                    style={{
                      position: 'relative',
                      border: selectedImage === image.id ? '3px solid blue' : '1px solid gray',
                      padding: '10px',
                      borderRadius: '4px',
                      transition: 'all 0.3s ease',
                      display: 'flex',
                      flexDirection: 'column',
                      minHeight: '400px'
                    }}
                    ref={el => imageRefs.current[image.id] = el}
                  >
                    {/* Action buttons above image */}
                    <div style={{
                      display: 'flex',
                      gap: '8px',
                      marginBottom: '10px'
                    }}>
                      <button
                        onClick={(e) => loadImageTags(image.id, e)}
                        style={{
                          background: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #ddd',
                          borderRadius: '6px',
                          padding: '6px 10px',
                          fontSize: '12px',
                          cursor: 'pointer',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                          transition: 'all 0.2s ease'
                        }}
                      >
                        Visual Tags
                      </button>
                      
                      <button
                        onClick={() => loadImageJson(image.id)}
                        style={{
                          background: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #ddd',
                          borderRadius: '6px',
                          padding: '6px 10px',
                          fontSize: '12px',
                          cursor: 'pointer',
                          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                          transition: 'all 0.2s ease'
                        }}
                      >
                        JSON Script
                      </button>
                    </div>
                    
                    <img 
                      src={image.url} 
                      alt={`Design ${image.id}`} 
                      style={{ 
                        width: '100%',
                        borderRadius: '2px',
                        cursor: 'pointer',
                        flexShrink: 0
                      }}
                      onClick={() => handleSelect(image.id)}
                    />

                    {/* Inline tag display */}
                    {showTagsByDefault && (
                      <InlineTagDisplay
                        key={`tags-${image.id}`}
                        tags={imageTagsMap[image.id] || []}
                        imageId={image.id}
                        onTagPreference={handleTagPreference}
                        preferences={derivedTagPreferences}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
            
            {/* Right column: Bubble chart */}
            <div style={{ flex: '0 0 50%', minWidth: '0' }}>
              <ConceptRefinementPanel
                sessionId={sessionId}
                stage="impression"
                images={images}
                selectedImage={selectedImage}
                onImageSelect={handleSelect}
                onTagClick={conceptTagHandlerRef}
                onTagPreferencesUpdate={handleConceptTagPreferencesUpdate}
              />
            </div>
          </div>

          {/* Continue Button */}
          <button
            onClick={handleSkipToSlider}
            disabled={!selectedImage || isLoading}
            style={{
              marginTop: '20px',
              padding: '12px 24px',
              backgroundColor: isLoading ? '#ccc' : (selectedImage ? '#28a745' : '#ccc'),
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: (!selectedImage || isLoading) ? 'not-allowed' : 'pointer',
              fontSize: '16px',
              fontWeight: '500',
              transition: 'background-color 0.3s ease'
            }}
          >
            {isLoading ? 'Processing...' : 'Continue to Slider Generation →'}
          </button>
          
          <button
            onClick={() => setStage('landing')}
            style={{
              marginTop: '20px',
              marginLeft: '10px',
              padding: '12px 24px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '16px'
            }}
          >
            ← Back to Sessions
          </button>
        </div>
      )}
      
      {/* Status messages */}
      {statusMessages.length > 0 && (
        <GenerationStatus messages={statusMessages} />
      )}

      {/* Tag sidebar drawer */}
      {showTagDrawer && (
        <TagSidebar
          tags={currentImageTags}
          onClose={() => setShowTagDrawer(false)}
          onTagPreference={handleTagPreference}
          imageId={currentImageId}
          position={drawerPosition}
          preferences={derivedTagPreferences}
        />
      )}

      {/* JSON panel */}
      {showJsonPanel && (
        <JsonPanel
          jsonData={currentImageJson}
          onClose={() => setShowJsonPanel(false)}
          imageId={currentImageId}
        />
      )}
    </div>
  );
}

export default App;
