import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ProgressBar from './components/ProgressBar';
import GenerationStatus from './components/GenerationStatus';
import TagSidebar from './components/TagSidebar';
import JsonPanel from './components/JsonPanel';
import InlineTagDisplay from './components/InlineTagDisplay';
import ConceptRefinementPanel from './components/ConceptRefinementPanel';

/**
 * Fisher-Yates shuffle algorithm to randomize array order
 */
const shuffleArray = (array) => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

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
  
  // Ranking/Evaluation state
  const [availableLocations, setAvailableLocations] = useState([]);
  const [currentRankingLocation, setCurrentRankingLocation] = useState(null);
  const [comparisonImages, setComparisonImages] = useState([]);
  const [rankings, setRankings] = useState({});  // {location: {1: filename, 2: filename, 3: filename}}
  const [currentRanking, setCurrentRanking] = useState({1: null, 2: null, 3: null, 4: null});
  const [sliderScores, setSliderScores] = useState({});  // {imageId: score (1-7)}
  const [sessionLogs, setSessionLogs] = useState([]);
  const [selectedSessionLog, setSelectedSessionLog] = useState(null);
  // Generation status: 'pending' | 'generating' | 'completed' | 'error'
  const [locationGenStatus, setLocationGenStatus] = useState({});
  const [isGeneratingLocations, setIsGeneratingLocations] = useState(false);
  // Image loading state for ranking view
  const [imagesLoaded, setImagesLoaded] = useState({});
  // Separate saving state to avoid UI refresh
  const [isSavingRanking, setIsSavingRanking] = useState(false);
  // Track if current ranking was just saved (for feedback)
  const [rankingSaved, setRankingSaved] = useState(false);
  
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
      addStatusMessage('Preparing evaluation...');
      
      // 1. Skip to slider (save weights, etc.)
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
        
        // 2. Get available locations and randomize their order
        const locsRes = await fetch('/api/eval/locations');
        let locations = [];
        if (locsRes.ok) {
          const locsData = await locsRes.json();
          // Shuffle locations to randomize display and generation order
          locations = shuffleArray(locsData.locations || []);
          setAvailableLocations(locations);
        }
        
        // Initialize generation status for all locations
        const initialStatus = {};
        locations.forEach(loc => {
          initialStatus[loc.name] = 'pending';
        });
        setLocationGenStatus(initialStatus);
        
        // 3. Generate initial location images (blocking)
        // Find the matching location name from available locations (for consistent capitalization)
        const explorationLoc = location; // from exploration (e.g., "bedroom")
        const matchingLoc = locations.find(loc => 
          loc.name.toLowerCase() === explorationLoc.toLowerCase()
        );
        const initialLocName = matchingLoc ? matchingLoc.name : explorationLoc;
        
        addStatusMessage(`Generating images for ${initialLocName}...`);
        setLocationGenStatus(prev => ({ ...prev, [initialLocName]: 'generating' }));
        
        const sliderRes = await fetch('/api/generate-slider', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            location: ''  // empty = initial location
          })
        });
        
        if (!sliderRes.ok) {
          throw new Error(`Failed to generate initial slider: ${sliderRes.status}`);
        }
        
        setLocationGenStatus(prev => ({ ...prev, [initialLocName]: 'completed' }));
        addStatusMessage(`Generated images for ${initialLocName}`);
        
        // 4. Initialize ranking session
        await fetch('/api/eval/init-ranking-session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_log: sessionId })
        });
        
        // 5. Set up for evaluation stage and load initial location images
        setSelectedSessionLog(sessionId);
        setCurrentRankingLocation(initialLocName);
        setStage('evaluation');
        
        // Load comparison images for initial location (use matched name for consistent casing)
        await loadComparisonImagesForSession(sessionId, initialLocName, true);
        
        // Generate all other locations immediately
        // We use the exploration selected image for style transfer, so no need to wait for ranking
        // Note: Use local 'locations' variable since state update is async
        if (locations.length > 1) {
          generateRemainingLocations(locations, initialLocName);
        }
      }
      
    } catch (error) {
      console.error('Error in handleSkipToSlider:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Preload images for a location to cache them in browser
  const preloadImagesForLocation = async (locName, isInitial = false) => {
    try {
      const sessionLog = selectedSessionLog || sessionId;
      const res = await fetch('/api/eval/get-comparison-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_log: sessionLog,
          location: locName,
          is_initial_round: isInitial
        })
      });
      
      if (res.ok) {
        const data = await res.json();
        // Preload each image by creating Image objects
        data.images.forEach(img => {
          const image = new Image();
          image.src = img.url;
        });
      }
    } catch (error) {
      console.error(`Error preloading images for ${locName}:`, error);
    }
  };
  
  // Background generation for remaining locations
  // Uses exploration selected image for style transfer (no need to wait for ranking)
  const generateRemainingLocations = async (locations, initialLoc) => {
    setIsGeneratingLocations(true);
    
    for (const loc of locations) {
      // Skip initial location (already generated) and Bedroom equivalent check
      const locName = loc.name;
      if (locName.toLowerCase() === initialLoc.toLowerCase() || 
          locName.toLowerCase().replace(/\s+/g, '_') === initialLoc.toLowerCase().replace(/\s+/g, '_')) {
        continue;
      }
      
      try {
        setLocationGenStatus(prev => ({ ...prev, [locName]: 'generating' }));
        addStatusMessage(`Generating images for ${locName}...`);
        
        const res = await fetch('/api/generate-slider', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            location: locName
          })
        });
        
        if (res.ok) {
          setLocationGenStatus(prev => ({ ...prev, [locName]: 'completed' }));
          addStatusMessage(`Generated images for ${locName}`);
          // Preload images in background for faster loading when user clicks
          preloadImagesForLocation(locName, false);
        } else {
          setLocationGenStatus(prev => ({ ...prev, [locName]: 'error' }));
          addStatusMessage(`Failed to generate images for ${locName}`);
        }
      } catch (error) {
        console.error(`Error generating ${locName}:`, error);
        setLocationGenStatus(prev => ({ ...prev, [locName]: 'error' }));
      }
    }
    
    setIsGeneratingLocations(false);
    addStatusMessage('All locations generated');
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

  // Custom stages for eval
  const evalStages = ['landing', 'impression', 'evaluation', 'slider_generation'];

  // ============== Ranking/Evaluation Functions ==============

  // Fetch session logs for selection
  const fetchSessionLogs = async () => {
    try {
      const res = await fetch('/api/eval/session-logs');
      if (res.ok) {
        const data = await res.json();
        setSessionLogs(data.session_logs || []);
      }
    } catch (error) {
      console.error('Failed to fetch session logs:', error);
      addStatusMessage(`Error loading session logs: ${error.message}`);
    }
  };

  // Fetch available locations from baseline_generic
  const fetchAvailableLocations = async () => {
    try {
      const res = await fetch('/api/eval/locations');
      if (res.ok) {
        const data = await res.json();
        // Shuffle locations to randomize display order
        setAvailableLocations(shuffleArray(data.locations || []));
      }
    } catch (error) {
      console.error('Failed to fetch locations:', error);
      addStatusMessage(`Error loading locations: ${error.message}`);
    }
  };

  // Load comparison images for a location (helper that takes session log)
  const loadComparisonImagesForSession = async (sessionLog, locationName, isInitialRound = false) => {
    if (!sessionLog) return;
    
    // Update title immediately for responsive UI
    setCurrentRankingLocation(locationName);
    setCurrentRanking({1: null, 2: null, 3: null, 4: null});
    setSliderScores({});  // Reset slider scores
    setRankingSaved(false);
    setImagesLoaded({});
    setComparisonImages([]); // Clear old images immediately
    
    setIsLoading(true);
    try {
      const res = await fetch('/api/eval/get-comparison-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_log: sessionLog,
          location: locationName,
          is_initial_round: isInitialRound
        })
      });
      
      if (!res.ok) {
        throw new Error(`Failed to load comparison images: ${res.status}`);
      }
      
      const data = await res.json();
      setComparisonImages(data.images);
      
    } catch (error) {
      console.error('Error loading comparison images:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Load comparison images for a location (uses selectedSessionLog or sessionId)
  // If location has been ranked before, restore the saved rankings
  const loadComparisonImages = async (locationName, isInitialRound = false) => {
    const sessionLog = selectedSessionLog || sessionId;
    if (!sessionLog) return;
    
    // Update title immediately for responsive UI
    setCurrentRankingLocation(locationName);
    setCurrentRanking({1: null, 2: null, 3: null, 4: null});
    setSliderScores({});  // Reset slider scores
    setRankingSaved(false);
    setImagesLoaded({});
    setComparisonImages([]); // Clear old images immediately
    
    setIsLoading(true);
    try {
      const res = await fetch('/api/eval/get-comparison-images', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_log: sessionLog,
          location: locationName,
          is_initial_round: isInitialRound
        })
      });
      
      if (!res.ok) {
        throw new Error(`Failed to load comparison images: ${res.status}`);
      }
      
      const data = await res.json();
      setComparisonImages(data.images);
      
      // Check if this location has saved rankings and restore them
      const savedRanking = rankings[locationName];
      if (savedRanking && Object.keys(savedRanking).length === 4) {
        // Map saved filenames back to image IDs
        const restoredRanking = {1: null, 2: null, 3: null, 4: null};
        const restoredScores = {};
        
        for (const [rank, rankData] of Object.entries(savedRanking)) {
          const savedFilename = rankData.image;
          const savedScore = rankData.score;
          
          // Find the image with this filename
          const matchingImage = data.images.find(img => img.filename === savedFilename);
          if (matchingImage) {
            restoredRanking[rank] = matchingImage.id;
            if (savedScore) {
              restoredScores[matchingImage.id] = savedScore;
            }
          }
        }
        
        setCurrentRanking(restoredRanking);
        setSliderScores(restoredScores);
        setRankingSaved(true);  // Show as already saved
        addStatusMessage(`Restored saved rankings for ${locationName}`);
      }
      
    } catch (error) {
      console.error('Error loading comparison images:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle ranking selection
  const handleRankingSelect = (imageId, rank) => {
    // Reset saved state when user changes ranking
    setRankingSaved(false);
    
    setCurrentRanking(prev => {
      const newRanking = { ...prev };
      // Remove this rank from any other image
      for (const key of Object.keys(newRanking)) {
        if (newRanking[key] === imageId) {
          newRanking[key] = null;
        }
      }
      // Set the new rank
      newRanking[rank] = imageId;
      return newRanking;
    });
  };

  // Save ranking for current location
  const saveCurrentRanking = async () => {
    if (!selectedSessionLog || !currentRankingLocation) return;
    
    // Find filenames and scores for each rank
    const rankingData = {};
    for (const [rank, imageId] of Object.entries(currentRanking)) {
      if (imageId) {
        const img = comparisonImages.find(i => i.id === imageId);
        const score = sliderScores[imageId];
        if (img) {
          rankingData[rank] = {
            image: img.filename,
            score: score
          };
        }
      }
    }
    
    if (Object.keys(rankingData).length !== 4) {
      addStatusMessage('Please rank all 4 images before saving');
      return;
    }
    
    // Check that all images have slider scores
    const missingScores = Object.entries(rankingData).filter(([rank, data]) => !data.score);
    if (missingScores.length > 0) {
      addStatusMessage('Please rate all 4 images with the preference slider before saving');
      return;
    }
    
    setIsSavingRanking(true);
    try {
      const res = await fetch('/api/eval/save-ranking', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_log: selectedSessionLog,
          location: currentRankingLocation,
          rankings: rankingData
        })
      });
      
      if (!res.ok) {
        throw new Error(`Failed to save ranking: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        // Update local rankings state
        setRankings(prev => ({
          ...prev,
          [currentRankingLocation]: rankingData
        }));
        addStatusMessage(`Saved ranking for ${currentRankingLocation}`);
        // Show saved feedback
        setRankingSaved(true);
        
        // Note: All locations are now generated immediately after initial location
        // since we use exploration selected image (not rank #1) for style transfer
      }
      
    } catch (error) {
      console.error('Error saving ranking:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsSavingRanking(false);
    }
  };

  // Generate remaining locations for resume sessions
  const generateRemainingLocationsForResume = async (sessionLogName, locations, generatedSet) => {
    setIsGeneratingLocations(true);
    // Helper to normalize location names (underscores <-> spaces)
    const normalizeName = (name) => name.toLowerCase().replace(/_/g, ' ');
    
    for (const loc of locations) {
      const locName = loc.name;
      // Skip already generated locations (use normalized names for comparison)
      if (generatedSet.has(normalizeName(locName))) {
        continue;
      }
      
      try {
        setLocationGenStatus(prev => ({ ...prev, [locName]: 'generating' }));
        addStatusMessage(`Generating images for ${locName}...`);
        
        const res = await fetch('/api/generate-slider', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionLogName,  // Use session log name as session_id
            location: locName
          })
        });
        
        if (res.ok) {
          setLocationGenStatus(prev => ({ ...prev, [locName]: 'completed' }));
          addStatusMessage(`Generated images for ${locName}`);
          // Preload images in background for faster loading when user clicks
          preloadImagesForLocation(locName, false);
        } else {
          setLocationGenStatus(prev => ({ ...prev, [locName]: 'error' }));
          addStatusMessage(`Failed to generate images for ${locName}`);
        }
      } catch (error) {
        console.error(`Error generating ${locName}:`, error);
        setLocationGenStatus(prev => ({ ...prev, [locName]: 'error' }));
      }
    }
    
    setIsGeneratingLocations(false);
    addStatusMessage('All locations generated');
  };

  // Initialize ranking session (supports resume of interrupted sessions)
  const initRankingSession = async (sessionLogName) => {
    setIsLoading(true);
    try {
      // Initialize (preserves existing rankings if present)
      await fetch('/api/eval/init-ranking-session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_log: sessionLogName })
      });
      
      // Load existing rankings if any
      const rankingsRes = await fetch(`/api/eval/get-rankings/${sessionLogName}`);
      let existingRankings = {};
      if (rankingsRes.ok) {
        const data = await rankingsRes.json();
        existingRankings = data.rankings || {};
        setRankings(existingRankings);
      }
      
      // Get available locations
      const locsRes = await fetch('/api/eval/locations');
      let locations = [];
      if (locsRes.ok) {
        const locsData = await locsRes.json();
        // Shuffle locations to randomize display order
        locations = shuffleArray(locsData.locations || []);
        setAvailableLocations(locations);
      }
      
      // Get which locations already have generated images (for resume)
      const genRes = await fetch(`/api/eval/session-locations/${sessionLogName}`);
      const genStatus = {};
      let generatedSet = new Set();
      // Helper to normalize location names (underscores <-> spaces)
      const normalizeName = (name) => name.toLowerCase().replace(/_/g, ' ');
      if (genRes.ok) {
        const genData = await genRes.json();
        // Normalize generated location names (folder names use underscores)
        generatedSet = new Set(genData.generated_locations.map(l => normalizeName(l)));
        locations.forEach(loc => {
          genStatus[loc.name] = generatedSet.has(normalizeName(loc.name)) ? 'completed' : 'pending';
        });
      }
      setLocationGenStatus(genStatus);
      
      setSelectedSessionLog(sessionLogName);
      setStage('evaluation');
      
      const rankedCount = Object.keys(existingRankings).length;
      const generatedCount = generatedSet.size;
      addStatusMessage(`Loaded session: ${sessionLogName} (${rankedCount} ranked, ${generatedCount} generated)`);
      
      // Auto-generate remaining locations if some are still pending
      const pendingLocations = locations.filter(loc => !generatedSet.has(normalizeName(loc.name)));
      if (pendingLocations.length > 0) {
        addStatusMessage(`Resuming generation for ${pendingLocations.length} remaining locations...`);
        // Generate remaining in background (don't await)
        generateRemainingLocationsForResume(sessionLogName, locations, generatedSet);
      }
      
    } catch (error) {
      console.error('Error initializing ranking session:', error);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Load session logs when on landing page
  useEffect(() => {
    if (stage === 'landing') {
      fetchSessionLogs();
    }
  }, [stage]);

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
          <h1 style={{ marginBottom: '30px', color: '#333' }}>Evaluation</h1>
          <p style={{ marginBottom: '30px', color: '#666', fontSize: '18px' }}>
            
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
              Enter your User ID:
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
              📂 Select a session
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

          {/* Session Logs Selection for Evaluation */}
          <div style={{
            marginTop: '30px',
            padding: '30px',
            backgroundColor: '#fff3cd',
            borderRadius: '12px',
            border: '2px solid #ffc107'
          }}>
            <h3 style={{ margin: '0 0 20px 0', color: '#333', fontSize: '18px' }}>
              📊 Evaluate Existing Session Logs
            </h3>
            <p style={{ color: '#666', marginBottom: '15px', fontSize: '14px' }}>
              Select a session log with generated images to rank and evaluate
            </p>

            {sessionLogs.length === 0 ? (
              <p style={{ color: '#666' }}>
                No session logs with generated images found.
              </p>
            ) : (
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
                gap: '15px'
              }}>
                {sessionLogs.map((log) => (
                  <button
                    key={log.name}
                    onClick={() => initRankingSession(log.name)}
                    disabled={isLoading}
                    style={{
                      padding: '15px',
                      backgroundColor: '#fff',
                      border: '2px solid #28a745',
                      borderRadius: '12px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease',
                      textAlign: 'left'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#e8f5e9';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = '#fff';
                      e.currentTarget.style.transform = 'translateY(0)';
                    }}
                  >
                    <div style={{ fontSize: '14px', fontWeight: '600', color: '#333', marginBottom: '6px' }}>
                      {log.descriptor || log.name}
                    </div>
                    <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>
                      {log.name}
                    </div>
                    <div style={{ fontSize: '11px', color: '#28a745' }}>
                      Locations: {log.locations.join(', ') || 'None'}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : stage === 'evaluation' ? (
        // ============== EVALUATION/RANKING STAGE ==============
        <div style={{ display: 'flex', gap: '20px', height: 'calc(100vh - 120px)' }}>
          {/* Left sidebar - Location navigator */}
          <div style={{
            width: '200px',
            flexShrink: 0,
            backgroundColor: '#f8f9fa',
            borderRadius: '12px',
            padding: '15px',
            overflowY: 'auto'
          }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', color: '#333' }}>
              Locations
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {availableLocations.map((loc, idx) => {
                const isRanked = rankings[loc.name] && Object.keys(rankings[loc.name]).length === 4;
                const isCurrent = currentRankingLocation === loc.name;
                // Check generation status
                const genStatus = locationGenStatus[loc.name] || 'pending';
                const isGenerated = genStatus === 'completed';
                const isGenerating = genStatus === 'generating';
                const isPending = genStatus === 'pending';
                // First location (Bedroom) is initial round
                const isInitialRound = loc.name.toLowerCase() === location.toLowerCase();
                
                // Determine button style based on status
                let bgColor = '#fff';
                let borderColor = '1px solid #dee2e6';
                let textColor = '#333';
                let cursor = 'pointer';
                
                if (isCurrent) {
                  bgColor = '#007bff';
                  borderColor = '2px solid #0056b3';
                  textColor = '#fff';
                } else if (isRanked) {
                  bgColor = '#d4edda';
                  borderColor = '2px solid #28a745';
                } else if (isGenerating) {
                  bgColor = '#fff3cd';
                  borderColor = '2px solid #ffc107';
                } else if (isPending) {
                  bgColor = '#e9ecef';
                  borderColor = '1px solid #ced4da';
                  textColor = '#6c757d';
                  cursor = 'not-allowed';
                }
                
                return (
                  <button
                    key={loc.name}
                    onClick={() => isGenerated && loadComparisonImages(loc.name, isInitialRound)}
                    disabled={isLoading || !isGenerated}
                    style={{
                      padding: '12px',
                      backgroundColor: bgColor,
                      color: textColor,
                      border: borderColor,
                      borderRadius: '8px',
                      cursor: cursor,
                      textAlign: 'left',
                      fontSize: '14px',
                      transition: 'all 0.2s ease',
                      opacity: isPending ? 0.6 : 1
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      {isRanked && <span style={{ color: '#28a745' }}>✓</span>}
                      {isGenerating && (
                        <span style={{ 
                          display: 'inline-block',
                          width: '14px',
                          height: '14px',
                          border: '2px solid #ffc107',
                          borderTop: '2px solid transparent',
                          borderRadius: '50%',
                          animation: 'spin 1s linear infinite'
                        }} />
                      )}
                      {isPending && <span style={{ color: '#6c757d' }}>○</span>}
                      {isGenerated && !isRanked && <span style={{ color: '#007bff' }}>●</span>}
                      <span>{loc.name}</span>
                    </div>
                  </button>
                );
              })}
            </div>
            
            {/* Generation status indicator */}
            {isGeneratingLocations && (
              <div style={{
                marginTop: '15px',
                padding: '10px',
                backgroundColor: '#fff3cd',
                borderRadius: '8px',
                textAlign: 'center',
                border: '1px solid #ffc107'
              }}>
                <div style={{ fontSize: '12px', color: '#856404' }}>Generating images...</div>
              </div>
            )}
            
            {/* Progress indicator */}
            <div style={{
              marginTop: '20px',
              padding: '10px',
              backgroundColor: '#e9ecef',
              borderRadius: '8px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '12px', color: '#666' }}>Progress</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#333' }}>
                {Object.keys(rankings).length} / {availableLocations.length}
              </div>
            </div>
            
            {/* Back button */}
            <button
              onClick={() => {
                setStage('landing');
                setSelectedSessionLog(null);
                setCurrentRankingLocation(null);
                setComparisonImages([]);
                setRankings({});
              }}
              style={{
                marginTop: '20px',
                width: '100%',
                padding: '10px',
                backgroundColor: '#6c757d',
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
          
          {/* Main area - Image comparison and ranking */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {!currentRankingLocation ? (
              <div style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: '#f8f9fa',
                borderRadius: '12px'
              }}>
                <div style={{ textAlign: 'center', color: '#666' }}>
                  <div style={{ fontSize: '48px', marginBottom: '20px' }}>📊</div>
                  <div style={{ fontSize: '18px' }}>Select a location from the sidebar to begin ranking</div>
                </div>
              </div>
            ) : (
              <>
                {/* Header */}
                <div style={{ marginBottom: '20px' }}>
                  <h2 style={{ margin: 0, color: '#333' }}>
                    Ranking: {adjective} {currentRankingLocation}
                  </h2>
                  <p style={{ color: '#666', margin: '5px 0 0 0' }}>
                    Rank the images based on how much you like each space, from most liked (1) to least liked (4).
                    <br />
                    Rank based on your personal liking, not on an objective definition or prior selections.
                  </p>
                </div>
                
                {/* Images display */}
                {isLoading ? (
                  <div style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <div style={{ fontSize: '18px', color: '#666' }}>Loading images...</div>
                  </div>
                ) : (
                  <div style={{ 
                    display: 'flex', 
                    gap: '20px', 
                    marginBottom: '20px',
                    alignItems: 'flex-start'
                  }}>
                    {comparisonImages.map((img, idx) => {
                      // Find what rank this image has
                      const imageRank = Object.entries(currentRanking).find(([rank, id]) => id === img.id)?.[0];
                      
                      return (
                        <div
                          key={img.id}
                          style={{
                            flex: '1 1 0',
                            display: 'flex',
                            flexDirection: 'column',
                            backgroundColor: '#fff',
                            borderRadius: '12px',
                            border: imageRank ? '3px solid #007bff' : '1px solid #dee2e6',
                            overflow: 'hidden',
                            maxWidth: '280px',
                            minWidth: 0
                          }}
                        >
                          {/* Image - Square aspect ratio using padding technique */}
                          <div style={{ 
                            position: 'relative',
                            width: '100%',
                            paddingBottom: '100%',
                            backgroundColor: '#f0f0f0',
                            flexShrink: 0,
                            overflow: 'hidden'
                          }}>
                            <img
                              src={img.url}
                              alt={`Option ${idx + 1}`}
                              style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                                opacity: imagesLoaded[img.id] ? 1 : 0,
                                transition: 'opacity 0.3s ease'
                              }}
                              onLoad={() => setImagesLoaded(prev => ({ ...prev, [img.id]: true }))}
                            />
                            {/* Loading indicator */}
                            {!imagesLoaded[img.id] && (
                              <div style={{
                                position: 'absolute',
                                top: '50%',
                                left: '50%',
                                transform: 'translate(-50%, -50%)',
                                color: '#666',
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
                                <span style={{ fontSize: '12px' }}>Loading...</span>
                              </div>
                            )}
                            {/* Rank badge */}
                            {imageRank && imagesLoaded[img.id] && (() => {
                              const rankColors = {
                                1: '#22c55e', // green
                                2: '#84cc16', // lime/yellow-green
                                3: '#f97316', // orange
                                4: '#ef4444'  // red
                              };
                              const ordinals = { 1: '1st', 2: '2nd', 3: '3rd', 4: '4th' };
                              return (
                                <div style={{
                                  position: 'absolute',
                                  top: '10px',
                                  left: '10px',
                                  width: '44px',
                                  height: '44px',
                                  backgroundColor: rankColors[imageRank] || '#007bff',
                                  borderRadius: '50%',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  color: 'white',
                                  fontSize: '14px',
                                  fontWeight: 'bold'
                                }}>
                                  {ordinals[imageRank] || imageRank}
                                </div>
                              );
                            })()}
                          </div>
                          
                          {/* Rank buttons */}
                          <div style={{
                            display: 'flex',
                            gap: '8px',
                            padding: '12px',
                            justifyContent: 'center',
                            backgroundColor: '#f8f9fa'
                          }}>
                            {[1, 2, 3, 4].map(rank => {
                              const rankColors = {
                                1: '#22c55e', // green
                                2: '#84cc16', // lime/yellow-green
                                3: '#f97316', // orange
                                4: '#ef4444'  // red
                              };
                              const ordinals = { 1: '1st', 2: '2nd', 3: '3rd', 4: '4th' };
                              const isSelected = currentRanking[rank] === img.id;
                              return (
                                <button
                                  key={rank}
                                  onClick={() => handleRankingSelect(img.id, rank)}
                                  style={{
                                    width: '44px',
                                    height: '44px',
                                    borderRadius: '50%',
                                    border: isSelected ? `3px solid ${rankColors[rank]}` : '2px solid #dee2e6',
                                    backgroundColor: isSelected ? rankColors[rank] : '#fff',
                                    color: isSelected ? '#fff' : '#333',
                                    fontSize: '12px',
                                    fontWeight: 'bold',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s ease'
                                  }}
                                >
                                  {ordinals[rank]}
                                </button>
                              );
                            })}
                          </div>
                          
                          {/* Preference Slider */}
                          <div style={{
                            padding: '20px 12px 12px 12px',
                            backgroundColor: '#f8f9fa',
                            borderTop: '1px solid #dee2e6'
                          }}>
                            <div style={{
                              fontSize: '13px',
                              color: '#666',
                              marginBottom: '8px',
                              textAlign: 'center',
                              fontWeight: '500'
                            }}>
                              This image matches my personal preference:
                            </div>
                            
                            {/* Slider track with dots */}
                            <div style={{ position: 'relative', marginBottom: '8px' }}>
                              {/* Labels above dots */}
                              <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginBottom: '8px',
                                paddingLeft: '12px',
                                paddingRight: '12px'
                              }}>
                                {[1, 2, 3, 4, 5, 6, 7].map(val => (
                                  <span key={val} style={{
                                    fontSize: '12px',
                                    fontWeight: sliderScores[img.id] === val ? 'bold' : 'normal',
                                    color: sliderScores[img.id] === val ? '#007bff' : '#666',
                                    width: '20px',
                                    textAlign: 'center'
                                  }}>
                                    {val}
                                  </span>
                                ))}
                              </div>
                              
                              {/* Track and dots */}
                              <div style={{
                                position: 'relative',
                                height: '40px',
                                display: 'flex',
                                alignItems: 'center',
                                paddingLeft: '12px',
                                paddingRight: '12px'
                              }}>
                                {/* Gray line */}
                                <div style={{
                                  position: 'absolute',
                                  left: '12px',
                                  right: '12px',
                                  height: '2px',
                                  backgroundColor: '#dee2e6',
                                  zIndex: 1
                                }} />
                                
                                {/* Clickable dots */}
                                <div style={{
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  width: '100%',
                                  position: 'relative',
                                  zIndex: 2
                                }}>
                                  {[1, 2, 3, 4, 5, 6, 7].map(val => {
                                    const sliderColors = {
                                      1: '#ef4444', // red
                                      2: '#f97316', // orange
                                      3: '#fb923c', // light orange
                                      4: '#facc15', // yellow
                                      5: '#a3e635', // lime
                                      6: '#4ade80', // light green
                                      7: '#22c55e'  // green
                                    };
                                    const isSelected = sliderScores[img.id] === val;
                                    const dotColor = sliderColors[val];
                                    return (
                                      <button
                                        key={val}
                                        onClick={() => {
                                          setSliderScores(prev => ({ ...prev, [img.id]: val }));
                                          setRankingSaved(false);
                                        }}
                                        style={{
                                          width: isSelected ? '24px' : '16px',
                                          height: isSelected ? '24px' : '16px',
                                          borderRadius: '50%',
                                          backgroundColor: dotColor,
                                          border: isSelected ? `3px solid ${dotColor}` : '2px solid transparent',
                                          cursor: 'pointer',
                                          padding: 0,
                                          transition: 'all 0.2s ease',
                                          boxShadow: isSelected ? `0 2px 6px ${dotColor}80` : 'none',
                                          opacity: isSelected ? 1 : 0.6
                                        }}
                                        onMouseEnter={(e) => {
                                          if (!isSelected) {
                                            e.currentTarget.style.transform = 'scale(1.2)';
                                            e.currentTarget.style.opacity = '1';
                                          }
                                        }}
                                        onMouseLeave={(e) => {
                                          e.currentTarget.style.transform = 'scale(1)';
                                          if (!isSelected) {
                                            e.currentTarget.style.opacity = '0.6';
                                          }
                                        }}
                                      />
                                    );
                                  })}
                                </div>
                              </div>
                            </div>
                            
                            {/* Scale labels */}
                            <div style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              fontSize: '11px',
                              color: '#666',
                              paddingLeft: '4px',
                              paddingRight: '4px'
                            }}>
                              <span>Strongly Disagree</span>
                              <span>Strongly Agree</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
                
                {/* Save button - centered */}
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <button
                    onClick={saveCurrentRanking}
                    disabled={(() => {
                      const allRanked = Object.values(currentRanking).filter(Boolean).length === 4;
                      const allScored = Object.values(currentRanking).filter(Boolean).every(imageId => sliderScores[imageId]);
                      return isSavingRanking || rankingSaved || !allRanked || !allScored;
                    })()}
                    style={{
                      padding: '12px 48px',
                      fontSize: '16px',
                      fontWeight: '500',
                      backgroundColor: (() => {
                        if (rankingSaved) return '#17a2b8';  // Blue when saved
                        const allRanked = Object.values(currentRanking).filter(Boolean).length === 4;
                        const allScored = Object.values(currentRanking).filter(Boolean).every(imageId => sliderScores[imageId]);
                        return allRanked && allScored ? '#28a745' : '#ccc';
                      })(),
                      color: 'white',
                      border: 'none',
                      borderRadius: '8px',
                      cursor: (() => {
                        const allRanked = Object.values(currentRanking).filter(Boolean).length === 4;
                        const allScored = Object.values(currentRanking).filter(Boolean).every(imageId => sliderScores[imageId]);
                        return rankingSaved || !allRanked || !allScored ? 'not-allowed' : 'pointer';
                      })(),
                      transition: 'background-color 0.3s ease'
                    }}
                  >
                    {isSavingRanking ? 'Saving...' : rankingSaved ? 'Saved ✓' : 'Save Ranking'}
                  </button>
                </div>
              </>
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
            <h2>Exploration Stage: {descriptor}</h2>
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

          {/* 4 images in a single row layout (no bubble chart) */}
          <div style={{ marginBottom: '20px' }}>
            <div style={{ 
              display: 'flex', 
              gap: '15px',
              width: '100%'
            }}>
              {images.map((image) => (
                <div
                  key={image.id}
                  style={{
                    flex: '1 1 0',
                    position: 'relative',
                    border: selectedImage === image.id ? '3px solid blue' : '1px solid gray',
                    padding: '10px',
                    borderRadius: '4px',
                    transition: 'all 0.3s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    minWidth: 0
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
          
          {/* Hidden ConceptRefinementPanel - still needed for concept system but not displayed */}
          <div style={{ display: 'none' }}>
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
            {isLoading ? 'Processing...' : 'Proceed to Next Stage →'}
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
