import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ProgressBar from './components/ProgressBar';
import GenerationStatus from './components/GenerationStatus';
import TagSidebar from './components/TagSidebar';
import JsonPanel from './components/JsonPanel';
import InlineTagDisplay from './components/InlineTagDisplay';
import ConceptRefinementPanel from './components/ConceptRefinementPanel';
import RefinementIterationControls from './components/RefinementIterationControls';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [stage, setStage] = useState('landing');

  // Refinement iteration state
  const [refinementRound, setRefinementRound] = useState(1);
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [adjectiveInput, setAdjectiveInput] = useState('');
  const [locationInput, setLocationInput] = useState('');
  const [statusMessages, setStatusMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [buttonColor, setButtonColor] = useState('#007bff');
  const [showTagDrawer, setShowTagDrawer] = useState(false);
  const [showJsonPanel, setShowJsonPanel] = useState(false);
  const [currentImageTags, setCurrentImageTags] = useState([]);
  const [currentImageId, setCurrentImageId] = useState(null);
  const [showTagsByDefault, setShowTagsByDefault] = useState(true);
  const [imageTagsMap, setImageTagsMap] = useState({}); // Store tags for all images
  const [currentImageJson, setCurrentImageJson] = useState(null);
  const [drawerPosition, setDrawerPosition] = useState({ top: 0, bottom: 0, left: 0 });
  const [userPreferences, setUserPreferences] = useState({
    selections: {},
    tags: {}
  });
  const [conceptTagPreferences, setConceptTagPreferences] = useState({}); // NEW: tag_id -> 'positive'|'negative'|null from concept system
  const [conceptSystemReady, setConceptSystemReady] = useState(false); // Track if concept system is initialized
  const [uploadedFolder, setUploadedFolder] = useState(null);
  const [validationError, setValidationError] = useState(null);
  
  // Slider generation state
  const [sliderRows, setSliderRows] = useState([]); // Array of {adjective, location, descriptor, images}
  const [sliderNewLocation, setSliderNewLocation] = useState('');
  const [sliderAdjective, setSliderAdjective] = useState(''); // Store the adjective from final_selection
  const [isGeneratingSlider, setIsGeneratingSlider] = useState(false);
  const [availableSessions, setAvailableSessions] = useState([]); // List of existing session folders
  const [selectedSessionPath, setSelectedSessionPath] = useState(''); // Selected session from dropdown
  const imageRefs = useRef({});
  const statusPollingRef = useRef({});
  const conceptTagHandlerRef = useRef(null); // Ref for concept-based tag handling

  // Add global styles for font
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

  // Reset concept system ready flag when stage changes
  useEffect(() => {
    console.log('[APP] Stage changed to:', stage, '- Resetting concept system ready flag');
    setConceptSystemReady(false);
    conceptTagHandlerRef.current = null;
  }, [stage]);

  // Load available sessions when upload page is shown
  useEffect(() => {
    if (stage === 'upload') {
      fetchAvailableSessions();
    }
  }, [stage]);

  // Load tags for all images when images change (skip for refinement stages)
  useEffect(() => {
    const isRefinementStage = stage === 'impression_refinement';
    if (images && images.length > 0 && sessionId && showTagsByDefault && !isRefinementStage) {
      loadAllImageTags(images);
    }
  }, [images, sessionId, showTagsByDefault, stage]);

  // Function to load tag weights for refinement images
  const loadImageWeights = async (imageId) => {
    try {
      // Parse image ID to get round and image index
      // Format for refinement: round_{N}_image_{idx} or image_{idx} or {stage}_refinement_{idx}_0
      console.log('[TAG WEIGHTS] Parsing image ID:', imageId);
      const parts = imageId.split('_');
      console.log('[TAG WEIGHTS] Parts:', parts);
      
      let roundNum = refinementRound; // Default to current round
      let imageIdx = 0;
      
      if (parts[0] === 'round') {
        roundNum = parseInt(parts[1]);
        imageIdx = parseInt(parts[3]);
      } else if (parts[0] === 'image') {
        imageIdx = parseInt(parts[1]);
      } else if (parts.includes('refinement')) {
        // Format like: impression_refinement_0_0
        const refinementIdx = parts.indexOf('refinement');
        if (refinementIdx >= 0 && parts.length > refinementIdx + 1) {
          imageIdx = parseInt(parts[refinementIdx + 1]);
        }
      }
      
      console.log('[TAG WEIGHTS] Parsed:', { imageId, roundNum, imageIdx, currentRefinementRound: refinementRound });
      
      // Load weights.json for this round
      const weightsPath = `/sessions/${sessionId}/${stage}/round_${roundNum}/weights.json`;
      console.log('[TAG WEIGHTS] Fetching:', weightsPath);
      const res = await fetch(weightsPath);
      
      if (!res.ok) throw new Error('Failed to load weights');
      
      const weightsData = await res.json();
      console.log('[TAG WEIGHTS] Loaded weights data with', weightsData.proposals.length, 'proposals');
      
      // Get weights for this specific image
      const imageWeights = weightsData.proposals[imageIdx];
      const conceptLabels = weightsData.concept_labels;
      
      if (!imageWeights) {
        throw new Error(`No weights found for image index ${imageIdx} (only ${weightsData.proposals.length} available)`);
      }
      
      // Create sorted list of concepts with weights > 0
      const weightsList = conceptLabels
        .map((label, idx) => ({ label, weight: imageWeights[idx] }))
        .filter(item => item.weight > 0)
        .sort((a, b) => b.weight - a.weight);
      
      // Format as readable text
      const formattedWeights = weightsList.length > 0
        ? weightsList.map((item, idx) => 
            `${idx + 1}. ${item.label}: ${(item.weight * 100).toFixed(2)}%`
          ).join('\n')
        : 'No concepts with weight > 0';
      
      setCurrentImageJson({ 
        tag_weights: formattedWeights,
        image_id: imageId,
        round: roundNum,
        total_concepts: weightsList.length,
        raw_data: { proposals: [imageWeights], concept_labels: conceptLabels }
      });
      setCurrentImageId(imageId);
      setShowJsonPanel(true);
      
      console.log('[TAG WEIGHTS] Loaded for image:', {
        imageId, roundNum, imageIdx, weightsCount: weightsList.length
      });
    } catch (error) {
      console.error('Failed to load weights:', error);
      addStatusMessage(`Failed to load tag weights: ${error.message}`);
    }
  };

  // Function to load JSON for an image
  const loadImageJson = async (imageId) => {
    // For refinement stages, show tag weights instead
    if (stage === 'impression_refinement') {
      return loadImageWeights(imageId);
    }

    try {
      // Determine the correct stage for the API call
      // Image ID format: impression_0_0_0, spatial_1_0_0, etc.
      const stageName = imageId.split('_')[0];
      const apiStage = stageName;

      const res = await fetch('/api/json', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          stage: apiStage,
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

  // Function to load tags for all images
  const loadAllImageTags = async (imagesList) => {
    if (!imagesList || imagesList.length === 0) return;
    
    try {
      const tagPromises = imagesList.map(async (image) => {
        // Determine the correct stage for the API call
        let apiStage = stage;
        if (stage === 'cumulative_tags') {
          const stageName = image.id.split('_')[0];
          apiStage = stageName;
        }

        try {
          const res = await fetch('/api/tags', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
              session_id: sessionId,
              stage: apiStage,
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

  // Function to load tags for an image (for the drawer)
  const loadImageTags = async (imageId, event) => {
    try {
      // Get the clicked image element's position
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

      // Determine the correct stage for the API call
      let apiStage = stage;
      if (stage === 'cumulative_tags') {
        // In cumulative tags mode, determine the actual stage from the image ID
        // Image ID format: impression_0_0_0, spatial_1_0_0, etc.
        const stageName = imageId.split('_')[0];
        apiStage = stageName;
      }

      const res = await fetch('/api/tags', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          stage: apiStage,
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

  // Callback for when concept system updates tag preferences
  const handleConceptTagPreferencesUpdate = useCallback((tagPrefs) => {
    console.log('[APP] ⭐ Concept tag preferences updated:', {
      count: Object.keys(tagPrefs).length,
      sample: Object.entries(tagPrefs).slice(0, 5).map(([k, v]) => ({ [k]: v }))
    });
    // Force new object reference to ensure React detects the change
    setConceptTagPreferences({ ...tagPrefs });
    
    // Mark system as ready when we receive preferences (means ConceptRefinementPanel is initialized)
    if (!conceptSystemReady) {
      console.log('[APP] ✅ Concept system is now ready');
      setConceptSystemReady(true);
    }
  }, [conceptSystemReady]);

  // Helper function to normalize tags for matching
  const normalizeTag = (tag) => {
    if (typeof tag !== 'string') return '';
    return tag.trim().toLowerCase();
  };

  // Function to handle tag preferences (now concept-based)
  const handleTagPreference = (tag, preference, imageId) => {
    // ===== DEBUG LOGS: Tag Click =====
    console.log('👆 [DEBUG] TAG CLICK (Concept-based):');
    console.log('  Tag:', tag);
    console.log('  Preference:', preference);
    console.log('  Image ID:', imageId);
    console.log('  Current Stage:', stage);
    console.log('  Is Refinement Stage:', stage === 'impression_refinement');
    console.log('  Concept Handler Available:', !!conceptTagHandlerRef.current);

    // Determine the actual stage
    const actualStage = imageId ? imageId.split('_')[0] : stage;
    
    // Build tag_id for concept system
    // Format: tag_{stage}_{imageId}_{index}
    // We need to find the index of this tag in the image's tags
    const imageTags = imageTagsMap[imageId] || [];
    
    // FIRST: Try exact match
    let tagIndex = imageTags.findIndex(t => t === tag);
    
    // SECOND: Try normalized match (handles whitespace, case differences)
    if (tagIndex === -1) {
      const normalizedTag = normalizeTag(tag);
      tagIndex = imageTags.findIndex(t => normalizeTag(t) === normalizedTag);
      
      if (tagIndex !== -1) {
        console.log('  ✅ Found tag with normalized match:', {
          original: tag,
          normalized: normalizedTag,
          matched: imageTags[tagIndex]
        });
      }
    }
    
    // THIRD: Try partial match (handles truncation)
    if (tagIndex === -1) {
      const normalizedTag = normalizeTag(tag);
      tagIndex = imageTags.findIndex(t => {
        const normalizedT = normalizeTag(t);
        return normalizedT.includes(normalizedTag) || normalizedTag.includes(normalizedT);
      });
      
      if (tagIndex !== -1) {
        console.log('  ✅ Found tag with partial match:', {
          original: tag,
          matched: imageTags[tagIndex]
        });
      }
    }
    
    console.log('  🔍 [DEBUG] Tag lookup:', {
      imageId,
      tag,
      imageTagsMapKeys: Object.keys(imageTagsMap),
      imageTagsForThisImage: imageTags,
      tagIndex,
      found: tagIndex !== -1
    });
    
    if (tagIndex === -1) {
      console.error('❌ Tag not found in imageTagsMap!', {
        tag,
        imageId,
        availableImages: Object.keys(imageTagsMap),
        tagsForImage: imageTags.slice(0, 5),
        allImageTags: Object.fromEntries(
          Object.entries(imageTagsMap).map(([k, v]) => [k, v.slice(0, 3)])
        )
      });
      
      // Show user feedback instead of silently failing
      addStatusMessage(`⚠️ Unable to set preference for "${tag}". Tag not found in current image data.`);
      return;
    }
    
    const tagId = `tag_${actualStage}_${imageId}_${tagIndex}`;
    console.log('  ✅ Tag ID:', tagId);
    console.log('  🔄 Checking concept handler status...');
    console.log('  conceptTagHandlerRef.current:', conceptTagHandlerRef.current);
    console.log('  Current conceptTagPreferences keys:', Object.keys(conceptTagPreferences).length);
    
    // Check if system is ready
    if (!conceptSystemReady || !conceptTagHandlerRef.current) {
      console.error('❌ [DEBUG] Concept system NOT READY!');
      console.error('  conceptSystemReady:', conceptSystemReady);
      console.error('  conceptTagHandlerRef.current:', !!conceptTagHandlerRef.current);
      console.error('  Current stage:', stage);
      console.error('  Is refinement stage:', stage === 'impression_refinement');

      if (stage === 'impression_refinement') {
        addStatusMessage('⚠️ Tag preferences are not available in refinement stages.');
      } else if (!conceptSystemReady) {
        addStatusMessage('⏳ Concept system is initializing... Please wait a moment and try again.');
      } else {
        addStatusMessage('⚠️ Preferences system not ready. Please wait a moment and try again.');
      }
      return;
    }
    
    // Call concept handler
    console.log('✅ [DEBUG] Calling concept handler with:', { tagId, preference });
    conceptTagHandlerRef.current(tagId, preference);
    console.log('✅ [DEBUG] Concept handler called successfully');
    
    // Show success feedback
    const prefEmoji = preference === 'positive' ? '👍' : '👎';
    addStatusMessage(`${prefEmoji} Set "${tag}" as ${preference}`);
  };

  // Derive UI preferences from concept system for tag components
  const derivedTagPreferences = useMemo(() => {
    // Determine the actual stage
    const actualStage = stage;

    console.log('[APP] Deriving tag preferences for UI:', {
      stage,
      actualStage,
      conceptTagPreferencesCount: Object.keys(conceptTagPreferences).length,
      imageTagsMapKeys: Object.keys(imageTagsMap).length,
      imageTagsMapContents: Object.fromEntries(
        Object.entries(imageTagsMap).map(([k, v]) => [k, `${v.length} tags`])
      ),
      sampleTagIds: Object.keys(conceptTagPreferences).slice(0, 5)
    });

    // Convert conceptTagPreferences (tag_id -> preference) to old format
    // Expected format: { tags: { [stage]: [ {tag, preference, source_image} ] }, currentStage }
    const derivedPrefs = {
      tags: {},
      currentStage: actualStage
    };

    // Convert tag_id format back to tag text + image_id
    let converted = 0;
    let skipped = { nullPreference: 0, invalidFormat: 0, noTagText: 0 };
    
    for (const [tagId, preference] of Object.entries(conceptTagPreferences)) {
      if (!preference) {
        skipped.nullPreference++;
        continue; // Skip null preferences
      }
      
      // tagId format: tag_{stage}_{imageId}_{index}
      const parts = tagId.split('_');
      if (parts.length < 4 || parts[0] !== 'tag') {
        console.warn('[APP] ⚠️ Invalid tag ID format:', tagId, parts);
        skipped.invalidFormat++;
        continue;
      }
      
      const tagStage = parts[1];
      // Reconstruct imageId (could have underscores)
      const tagIndex = parseInt(parts[parts.length - 1]);
      const imageId = parts.slice(2, parts.length - 1).join('_');
      
      // Get the actual tag text from imageTagsMap
      const imageTags = imageTagsMap[imageId] || [];
      const tagText = imageTags[tagIndex];
      
      if (!tagText) {
        console.error('[APP] ❌ Tag text not found:', {
          tagId,
          tagId_parts: parts,
          reconstructedImageId: imageId,
          tagIndex,
          tagIndexType: typeof tagIndex,
          isNaN_tagIndex: isNaN(tagIndex),
          imageIdExistsInMap: imageId in imageTagsMap,
          availableTagsForThisImage: imageTags,
          imageTagsLength: imageTags.length,
          allImageTagsMapKeys: Object.keys(imageTagsMap).sort(),
          // Show first few tags of each image for debugging
          firstTagsPerImage: Object.fromEntries(
            Object.entries(imageTagsMap).map(([k, v]) => [k, v.slice(0, 3)])
          )
        });
        skipped.noTagText++;
        continue;
      }
      
      if (!derivedPrefs.tags[tagStage]) {
        derivedPrefs.tags[tagStage] = [];
      }
      
      derivedPrefs.tags[tagStage].push({
        tag: tagText,
        preference: preference,
        source_image: imageId
      });
      
      converted++;
      console.log('[APP] ✅ Converted tag preference:', {
        tagId,
        tagStage,
        imageId,
        tagIndex,
        tagText,
        preference
      });
    }
    
    console.log('[APP] 📊 Conversion summary:', {
      total: Object.keys(conceptTagPreferences).length,
      converted,
      skipped,
      skippedDetails: {
        nullPreference: `${skipped.nullPreference} tags with null preference`,
        invalidFormat: `${skipped.invalidFormat} tags with invalid format`,
        noTagText: `${skipped.noTagText} tags where text couldn't be found`
      }
    });

    console.log('[APP] Derived tag preferences:', {
      stages: Object.keys(derivedPrefs.tags),
      counts: Object.fromEntries(
        Object.entries(derivedPrefs.tags).map(([k, v]) => [k, v.length])
      ),
      sample: derivedPrefs.tags[actualStage]?.slice(0, 3)
    });

    return derivedPrefs;
  }, [conceptTagPreferences, imageTagsMap, stage]);

  // Function to add status message
  const addStatusMessage = useCallback((message) => {
    setStatusMessages(prev => [...prev, message]);
    // Auto-scroll to bottom of status window
    const statusElement = document.querySelector('.status-messages');
    if (statusElement) {
      statusElement.scrollTop = statusElement.scrollHeight;
    }
  }, []);

  // Status polling function
  const pollStatus = useCallback(async (sid) => {
    try {
      const res = await fetch(`/api/status/${sid}`);
      if (!res.ok) return;
      
      const data = await res.json();
      if (data.messages && data.messages.length > 0) {
        data.messages.forEach(msg => addStatusMessage(msg));
      }
      
      // If generation is complete, stop polling
      if (data.status === 'complete') {
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error polling status:', error);
      return false;
    }
  }, [addStatusMessage]);

  // Start polling function
  const startStatusPolling = () => {
    if (!sessionId) return;
    
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`/api/status/${sessionId}`);
        if (response.ok) {
          const newMessages = await response.text();
          if (newMessages.trim()) {
            const lines = newMessages.trim().split('\n');
            lines.forEach(line => {
              if (line.trim() && !statusMessages.some(msg => msg.includes(line.trim()))) {
                addStatusMessage(line.trim());
              }
            });
          }
        }
      } catch (error) {
        console.log('Status polling error:', error);
      }
    }, 1000);
    
    // Store interval for cleanup
    if (statusPollingRef.current) {
      statusPollingRef.current.statusPolling = pollInterval;
    }
  };

  const handleSubmit = async ({ adjective, location, descriptor }) => {
    try {
      setIsLoading(true);
      addStatusMessage(`Starting fast sequential generation for: ${descriptor}`);
      
      const res = await fetch('/api/generate-fast', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ adjective, location, descriptor }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      setSessionId(data.session_id);
      setStage(data.stage);
      setImages(data.images);
      
      // Start polling for status messages
      startStatusPolling();
    } catch (error) {
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };




  const handleSelect = (imageId) => {
    setSelectedImage(imageId);
    // Update selections in preferences
    setUserPreferences(prev => ({
      ...prev,
      selections: {
        ...prev.selections,
        [stage]: imageId
      }
    }));
  };

  const handleContinue = async (historicalItem = null) => {
    try {
      setIsLoading(true);
      setButtonColor('#4CAF50'); // Change to green when continuing

      // Load tag weights for the selected image if in refinement stage
      let tagWeights = null;
      let weights = null;
      let roundNum = refinementRound;
      let imageIdx = 0;
      let imagePath = '';
      let isHistorical = false;
      
      if (stage.endsWith('_refinement')) {
        // Check if this is a historical selection
        if (historicalItem && historicalItem.weights) {
          console.log('[SaveFinalSelection] Using historical selection:', historicalItem.label);
          weights = historicalItem.weights;
          roundNum = historicalItem.round || refinementRound;
          imagePath = historicalItem.imageUrl || '';
          isHistorical = true;
        } else if (selectedImage) {
          // Parse image ID to get round and image index
          const parts = selectedImage.split('_');
          
          if (parts[0] === 'round') {
            roundNum = parseInt(parts[1]);
            imageIdx = parseInt(parts[3]);
          } else if (parts[0] === 'image') {
            imageIdx = parseInt(parts[1]);
          } else if (parts.includes('refinement')) {
            const refinementIdx = parts.indexOf('refinement');
            if (refinementIdx >= 0 && parts.length > refinementIdx + 1) {
              imageIdx = parseInt(parts[refinementIdx + 1]);
            }
          }
          
          // Load weights.json for this round
          const weightsPath = `/sessions/${sessionId}/${stage}/round_${roundNum}/weights.json`;
          try {
            const weightsRes = await fetch(weightsPath);
            
            if (weightsRes.ok) {
              const weightsData = await weightsRes.json();
              weights = weightsData.proposals[imageIdx];
              const conceptLabels = weightsData.concept_labels;
              imagePath = `${stage}/round_${roundNum}/image_${imageIdx}.png`;
              
              if (weights && conceptLabels) {
                // Create dictionary: tag name -> weight
                tagWeights = {};
                conceptLabels.forEach((label, idx) => {
                  if (weights[idx] > 0) {
                    tagWeights[label] = weights[idx];
                  }
                });
                console.log(`✅ Loaded ${Object.keys(tagWeights).length} tag weights for selection`);
              }
            }
          } catch (error) {
            console.error('Failed to load tag weights:', error);
          }
        }
        
        // Save final selection if we have weights
        if (weights) {
          const baseStage = stage.replace('_refinement', '');
          try {
            const saveRes = await fetch('/api/save-final-selection', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                session_id: sessionId,
                stage: baseStage,
                weights: weights,
                image_path: imagePath,
                round_number: roundNum,
                is_historical: isHistorical
              })
            });
            
            if (saveRes.ok) {
              const saveData = await saveRes.json();
              console.log(`✅ Final selection saved: ${saveData.message}`);
              addStatusMessage(`✅ Final selection saved with ${weights.length} concept weights`);
              
              // Navigate to slider generation stage
              setStage('slider_generation');
              setSliderRows([]);
              setIsLoading(false);
              return; // Exit early - don't call feedback endpoint
            } else {
              console.error('Failed to save final selection:', await saveRes.text());
            }
          } catch (error) {
            console.error('Error saving final selection:', error);
          }
        }
      }

      // Determine the selected image ID - use historical item's imageId if available
      const finalSelectedImageId = (historicalItem && historicalItem.imageId) 
        ? historicalItem.imageId 
        : selectedImage;
      
      console.log('[Feedback] Sending request:', {
        session_id: sessionId,
        stage,
        selected_image_id: finalSelectedImageId,
        has_preferences: !!userPreferences,
        has_tag_weights: !!tagWeights
      });

      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          stage,
          selected_image_id: finalSelectedImageId,
          preferences: userPreferences || {}, // Ensure preferences is always an object
          tag_weights: tagWeights // Send tag weights if available
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();
      if (data.next_stage) {
        setStage(data.next_stage);
        setImages(data.images);
        setSelectedImage(null);
        setButtonColor('#007bff'); // Reset color for next stage
        setShowTagDrawer(false); // Close sidebar when moving to next stage
        setShowJsonPanel(false); // Close JSON panel when moving to next stage

        // Reset refinement round when entering impression_refinement stage
        if (data.next_stage === 'impression_refinement') {
          setRefinementRound(1);
        }
      }
    } catch (error) {
      addStatusMessage(`Error: ${error.message}`);
      setButtonColor('#f44336'); // Change to red on error
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputSubmit = () => {
    if (adjectiveInput.trim() && locationInput.trim()) {
      const descriptor = `${adjectiveInput.trim()} ${locationInput.trim()}`;
      handleSubmit({ 
        adjective: adjectiveInput.trim(), 
        location: locationInput.trim(), 
        descriptor 
      });
      setAdjectiveInput('');
      setLocationInput('');
    }
  };

  // Slider generation function
  const generateSlider = async (location = '') => {
    if (!sessionId) return;
    
    try {
      setIsGeneratingSlider(true);
      addStatusMessage(`Generating slider for ${location || 'original location'}...`);
      
      const res = await fetch('/api/generate-slider', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          location: location
        })
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.success) {
        // Store the adjective for future generations
        setSliderAdjective(data.adjective);
        
        // Add new slider row
        setSliderRows(prev => [...prev, {
          adjective: data.adjective,
          location: data.location,
          descriptor: data.descriptor,
          images: data.images
        }]);
        
        addStatusMessage(`✅ Generated slider for "${data.descriptor}"`);
        setSliderNewLocation(''); // Clear input after generation
      }
    } catch (error) {
      addStatusMessage(`Error generating slider: ${error.message}`);
    } finally {
      setIsGeneratingSlider(false);
    }
  };

  // Auto-generate slider when entering slider_generation stage
  useEffect(() => {
    if (stage === 'slider_generation' && sliderRows.length === 0 && !isGeneratingSlider) {
      generateSlider(''); // Generate with original location
    }
  }, [stage]);

  // Cleanup polling intervals when component unmounts or stage changes
  useEffect(() => {
    return () => {
      if (statusPollingRef.current?.statusPolling) {
        clearInterval(statusPollingRef.current.statusPolling);
      }
    };
  }, []);

  // Folder upload handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  // Fetch available sessions from backend
  const fetchAvailableSessions = async () => {
    try {
      const res = await fetch('/api/list-sessions');
      if (!res.ok) {
        throw new Error('Failed to fetch sessions');
      }
      const data = await res.json();
      setAvailableSessions(data.sessions || []);
    } catch (error) {
      console.error('Error fetching sessions:', error);
      setAvailableSessions([]);
    }
  };

  // Load selected session from dropdown
  const handleLoadSelectedSession = async () => {
    if (!selectedSessionPath) {
      setValidationError('Please select a session');
      return;
    }

    try {
      setIsLoading(true);
      addStatusMessage('Loading session data...');

      // Load session metadata
      const res = await fetch('/api/load-stage-data', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_path: selectedSessionPath,
          stage: 'impression'
        })
      });

      if (!res.ok) {
        throw new Error(`Failed to load session: ${res.status}`);
      }

      const data = await res.json();

      // Set session data
      setSessionId(selectedSessionPath);
      setUserPreferences(data.preferences || {});
      setImages(data.images);
      setStage('impression');

      addStatusMessage('Session loaded successfully! Review images and continue to refinement.');

    } catch (error) {
      setValidationError(`Failed to load session: ${error.message}`);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFolderDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const items = e.dataTransfer.items;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        const item = items[i].webkitGetAsEntry();
        if (item && item.isDirectory) {
          processUploadedFolder(item);
          break;
        }
      }
    }
  };

  const handleFolderSelect = (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      // Create a mock directory entry from files
      const folderName = files[0].webkitRelativePath.split('/')[0];
      const mockFolder = {
        name: folderName,
        files: files
      };
      processUploadedFolder(mockFolder);
    }
  };

  const processUploadedFolder = async (folderEntry) => {
    setValidationError(null);
    setUploadedFolder(null);
    
    try {
      // For mock folder from file input
      if (folderEntry.files) {
        await validateAndLoadFolder(folderEntry);
        return;
      }
      
      // For actual folder drop
      const files = [];
      await readDirectoryRecursively(folderEntry, files);
      
      const mockFolder = {
        name: folderEntry.name,
        files: files
      };
      
      await validateAndLoadFolder(mockFolder);
    } catch (error) {
      setValidationError(`Failed to process folder: ${error.message}`);
    }
  };

  const readDirectoryRecursively = (dirEntry, files) => {
    return new Promise((resolve) => {
      const reader = dirEntry.createReader();
      reader.readEntries((entries) => {
        let pending = entries.length;
        if (pending === 0) {
          resolve();
          return;
        }
        
        entries.forEach((entry) => {
          if (entry.isFile) {
            entry.file((file) => {
              files.push({
                name: file.name,
                path: entry.fullPath,
                file: file
              });
              pending--;
              if (pending === 0) resolve();
            });
          } else if (entry.isDirectory) {
            readDirectoryRecursively(entry, files).then(() => {
              pending--;
              if (pending === 0) resolve();
            });
          }
        });
      });
    });
  };

  const validateAndLoadFolder = async (folderData) => {
    // Validate required structure (simplified for impression-only)
    // Only impression.json is required; preferences.json is optional (created during exploration)
    const requiredFiles = [
      'impression/impression.json'
    ];
    
    const fileMap = {};
    const files = folderData.files || [];
    
    // Build file map
    files.forEach(file => {
      let path;
      if (file.webkitRelativePath) {
        // From file input
        path = file.webkitRelativePath.split('/').slice(1).join('/');
      } else if (file.path) {
        // From folder drop
        path = file.path.startsWith('/') ? file.path.slice(1) : file.path;
      }
      
      if (path) {
        fileMap[path] = file;
      }
    });
    
    // Check for required files
    const missingFiles = [];
    for (const requiredFile of requiredFiles) {
      if (!fileMap[requiredFile]) {
        missingFiles.push(requiredFile);
      }
    }
    
    if (missingFiles.length > 0) {
      setValidationError(`Missing required files: ${missingFiles.join(', ')}`);
      return;
    }
    
    // Validation passed
    setUploadedFolder(folderData);
    addStatusMessage(`Session folder "${folderData.name}" loaded successfully`);
  };

  const handleProceedWithUpload = async () => {
    if (!uploadedFolder) return;
    
    try {
      setIsLoading(true);
      addStatusMessage('Loading session data...');
      
      // Create FormData to upload folder
      const formData = new FormData();
      const files = uploadedFolder.files || [];
      
      files.forEach(file => {
        const actualFile = file.file || file;
        const path = file.webkitRelativePath || file.path || file.name;
        formData.append('files', actualFile, path);
      });
      
      formData.append('folderName', uploadedFolder.name);
      
      const res = await fetch('/api/upload-session', {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        throw new Error(`Upload failed: ${res.status}`);
      }
      
      const data = await res.json();

      // Set session data and load impression stage
      setSessionId(data.session_id);
      setUserPreferences(data.preferences);

      // Load impression stage data
      const loadRes = await fetch('/api/load-stage-data', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_path: data.session_id,
          stage: 'impression'
        }),
      });

      const loadData = await loadRes.json();
      setImages(loadData.images);
      setStage('impression');

      addStatusMessage('Session loaded successfully! Review images and continue to refinement.');
      
    } catch (error) {
      setValidationError(`Upload failed: ${error.message}`);
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      padding: '20px',
      fontFamily: 'SF Pro Text, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif'
    }}>
      
      {/* Progress bar for all stages */}
      <ProgressBar currentStage={stage} />
      
      {stage === 'landing' ? (
        <div style={{ textAlign: 'center', maxWidth: '800px', margin: '0 auto' }}>
          <h1 style={{ marginBottom: '30px', color: '#333' }}>Environment Design Generator</h1>
          <p style={{ marginBottom: '50px', color: '#666', fontSize: '18px' }}>
            Choose how you want to start your environment design process:
          </p>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px', marginBottom: '40px' }}>
            {/* Run Pipeline Button */}
            <div
              onClick={() => setStage('input')}
              style={{
                border: '2px solid #007bff',
                borderRadius: '12px',
                padding: '30px 20px',
                backgroundColor: '#f0f8ff',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
              onMouseEnter={e => {
                e.currentTarget.style.backgroundColor = '#e6f3ff';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.backgroundColor = '#f0f8ff';
                e.currentTarget.style.transform = 'translateY(0px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
              }}
            >
              <h3 style={{ margin: '0 0 12px 0', color: '#007bff', fontSize: '20px' }}>
                🚀 Run Complete Pipeline
              </h3>
              <p style={{ margin: '0', color: '#333', fontSize: '14px', lineHeight: '1.5' }}>
                Start from scratch: Impression → Spatial → Objects → Ambient → Final
              </p>
            </div>

            {/* Test Final Stage Button */}
            <div
              onClick={() => setStage('upload')}
              style={{
                border: '2px solid #28a745',
                borderRadius: '12px',
                padding: '30px 20px',
                backgroundColor: '#f0fff0',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
              onMouseEnter={e => {
                e.currentTarget.style.backgroundColor = '#e6ffe6';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.backgroundColor = '#f0fff0';
                e.currentTarget.style.transform = 'translateY(0px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
              }}
            >
              <h3 style={{ margin: '0 0 12px 0', color: '#28a745', fontSize: '20px' }}>
                🎯 Load Existing Session
              </h3>
              <p style={{ margin: '0', color: '#333', fontSize: '14px', lineHeight: '1.5' }}>
                Select from existing sessions or upload a session folder
              </p>
            </div>
          </div>
        </div>
      ) : stage === 'upload' ? (
        <div style={{ maxWidth: '800px', margin: '0 auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '30px' }}>
            <button
              onClick={() => setStage('landing')}
              style={{
                padding: '8px 16px',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: 'pointer',
                marginRight: '20px'
              }}
            >
              ← Back
            </button>
            <h2 style={{ margin: '0', color: '#333' }}>Load Session</h2>
          </div>

          {/* Dropdown to select existing session */}
          <div style={{
            marginBottom: '30px',
            padding: '30px',
            backgroundColor: '#e8f4f8',
            borderRadius: '12px',
            border: '2px solid #007bff'
          }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#333', fontSize: '18px' }}>
              📂 Load Existing Session
            </h3>
            <p style={{ margin: '0 0 20px 0', color: '#666', fontSize: '14px' }}>
              Select a session from your existing sessions folder
            </p>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'flex-start' }}>
              <select
                value={selectedSessionPath}
                onChange={(e) => setSelectedSessionPath(e.target.value)}
                disabled={isLoading}
                style={{
                  flex: 1,
                  padding: '12px',
                  fontSize: '16px',
                  borderRadius: '6px',
                  border: '1px solid #ccc',
                  backgroundColor: 'white',
                  cursor: isLoading ? 'not-allowed' : 'pointer'
                }}
              >
                <option value="">-- Select a session --</option>
                {availableSessions.map(session => (
                  <option key={session.path} value={session.path}>
                    {session.name} ({session.timestamp}) - {session.stage_count} stage(s)
                  </option>
                ))}
              </select>
              <button
                onClick={handleLoadSelectedSession}
                disabled={!selectedSessionPath || isLoading}
                style={{
                  padding: '12px 24px',
                  backgroundColor: (!selectedSessionPath || isLoading) ? '#ccc' : '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: (!selectedSessionPath || isLoading) ? 'not-allowed' : 'pointer',
                  fontSize: '16px',
                  whiteSpace: 'nowrap'
                }}
              >
                {isLoading ? 'Loading...' : 'Load Session'}
              </button>
            </div>
          </div>

          {/* Divider */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            margin: '30px 0',
            color: '#999'
          }}>
            <div style={{ flex: 1, height: '1px', backgroundColor: '#ddd' }}></div>
            <span style={{ padding: '0 20px', fontSize: '14px' }}>OR</span>
            <div style={{ flex: 1, height: '1px', backgroundColor: '#ddd' }}></div>
          </div>

          {/* Upload new session */}
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ margin: '0 0 15px 0', color: '#333', fontSize: '18px' }}>
              📤 Upload New Session
            </h3>
          </div>

          <div
            onDrop={handleFolderDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            style={{
              border: '3px dashed #007bff',
              borderRadius: '12px',
              padding: '60px 40px',
              textAlign: 'center',
              backgroundColor: '#f8f9fa',
              marginBottom: '30px',
              transition: 'all 0.3s ease'
            }}
          >
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>📁</div>
            <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>Drop Session Folder Here</h3>
            <p style={{ margin: '0', color: '#666', fontSize: '16px' }}>
              Or click to browse and select your session folder
            </p>
            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              onChange={handleFolderSelect}
              style={{ display: 'none' }}
              id="folder-input"
            />
            <button
              onClick={() => document.getElementById('folder-input').click()}
              style={{
                marginTop: '20px',
                padding: '12px 24px',
                backgroundColor: '#007bff',
                color: 'white',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '16px'
              }}
            >
              Browse Folder
            </button>
          </div>
          
          {validationError && (
            <div style={{
              padding: '15px',
              backgroundColor: '#f8d7da',
              border: '1px solid #f5c6cb',
              borderRadius: '6px',
              color: '#721c24',
              marginBottom: '20px'
            }}>
              <strong>Validation Error:</strong> {validationError}
            </div>
          )}
          
          {uploadedFolder && !validationError && (
            <div style={{
              padding: '20px',
              backgroundColor: '#d4edda',
              border: '1px solid #c3e6cb',
              borderRadius: '6px',
              marginBottom: '20px'
            }}>
              <h4 style={{ margin: '0 0 10px 0', color: '#155724' }}>
                ✅ Session Folder Loaded Successfully
              </h4>
              <p style={{ margin: '0 0 15px 0', color: '#155724' }}>
                <strong>Folder:</strong> {uploadedFolder.name}
              </p>
              <button
                onClick={handleProceedWithUpload}
                disabled={isLoading}
                style={{
                  padding: '12px 24px',
                  backgroundColor: isLoading ? '#ccc' : '#28a745',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                  fontSize: '16px'
                }}
              >
                {isLoading ? 'Loading...' : 'Proceed to Mode Selection'}
              </button>
            </div>
          )}
        </div>
      ) : stage === 'slider_generation' ? (
        <div>
          <h2>Semantic Slider Generation</h2>
          
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
          
          {/* Slider rows */}
          {sliderRows.map((row, rowIndex) => (
            <div key={rowIndex} style={{
              marginBottom: '30px',
              padding: '20px',
              backgroundColor: '#f8f9fa',
              borderRadius: '12px',
              border: '1px solid #dee2e6'
            }}>
              {/* Alpha scale header - only show on first row */}
              {rowIndex === 0 && (
                <div style={{ marginBottom: '15px' }}>
                  {/* Alpha labels */}
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: '8px',
                    padding: '0 10px'
                  }}>
                    <span style={{ fontSize: '14px', color: '#666' }}>alpha = 0</span>
                    <span style={{ fontSize: '14px', color: '#666' }}>0.25</span>
                    <span style={{ fontSize: '14px', color: '#666' }}>0.50</span>
                    <span style={{ fontSize: '14px', color: '#666' }}>0.75</span>
                    <span style={{ fontSize: '14px', color: '#666' }}>alpha = 1</span>
                  </div>
                  
                  {/* Arrow line with dots */}
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0 10px',
                    marginBottom: '5px'
                  }}>
                    <span style={{ fontWeight: 'bold', color: '#333', fontSize: '16px' }}>generic</span>
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
                        borderTop: '6px solid transparent',
                        borderBottom: '6px solid transparent',
                        borderRight: '10px solid #666'
                      }} />
                      {/* Line */}
                      <div style={{ 
                        flex: 1, 
                        height: '2px', 
                        backgroundColor: '#666',
                        position: 'relative'
                      }}>
                        {/* Dots */}
                        {[0.25, 0.5, 0.75].map((pos, i) => (
                          <div key={i} style={{
                            position: 'absolute',
                            left: `${pos * 100}%`,
                            top: '50%',
                            transform: 'translate(-50%, -50%)',
                            width: '10px',
                            height: '10px',
                            backgroundColor: '#666',
                            borderRadius: '50%'
                          }} />
                        ))}
                      </div>
                      {/* Right arrow */}
                      <div style={{
                        width: 0,
                        height: 0,
                        borderTop: '6px solid transparent',
                        borderBottom: '6px solid transparent',
                        borderLeft: '10px solid #666'
                      }} />
                    </div>
                    <span style={{ fontWeight: 'bold', color: '#333', fontSize: '16px' }}>personalized</span>
                  </div>
                </div>
              )}
              
              {/* Images row with label */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                {/* 5 images */}
                <div style={{ display: 'flex', gap: '10px', flex: 1 }}>
                  {row.images.map((img, imgIndex) => (
                    <div key={imgIndex} style={{ 
                      flex: 1,
                      aspectRatio: '1',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}>
                      <img 
                        src={img.url} 
                        alt={`Alpha ${img.alpha}`}
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
                    {row.adjective}
                  </div>
                  <div style={{ 
                    fontSize: '16px', 
                    color: '#666',
                    textTransform: 'capitalize'
                  }}>
                    {row.location}
                  </div>
                </div>
              </div>
            </div>
          ))}
          
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
        </div>
      ) : stage === 'input' ? (
        <div>
          <h2>Enter Design Description</h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '20px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500', color: '#374151' }}>
                Adjective
              </label>
              <input
                type="text"
                value={adjectiveInput}
                onChange={(e) => setAdjectiveInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleInputSubmit();
                  }
                }}
                placeholder="e.g., cozy, modern, rustic..."
                style={{
                  width: '100%',
                  padding: '10px',
                  fontSize: '16px',
                  borderRadius: '4px',
                  border: '1px solid #ccc'
                }}
                disabled={isLoading}
              />
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontWeight: '500', color: '#374151' }}>
                Location
              </label>
              <input
                type="text"
                value={locationInput}
                onChange={(e) => setLocationInput(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleInputSubmit();
                  }
                }}
                placeholder="e.g., space, bedroom, kitchen..."
                style={{
                  width: '100%',
                  padding: '10px',
                  fontSize: '16px',
                  borderRadius: '4px',
                  border: '1px solid #ccc'
                }}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleInputSubmit}
              disabled={!(adjectiveInput.trim() && locationInput.trim()) || isLoading}
              style={{
                padding: '10px 20px',
                fontSize: '16px',
                backgroundColor: isLoading ? '#ccc' : ((adjectiveInput.trim() && locationInput.trim()) ? buttonColor : '#ccc'),
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: (!(adjectiveInput.trim() && locationInput.trim()) || isLoading) ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.3s ease'
              }}
            >
              {isLoading ? 'Generating...' : 'Send'}
            </button>
          </div>
        </div>
      ) : (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2>Current Stage: {stage.includes('_refinement') ? stage.replace('_refinement', '').charAt(0).toUpperCase() + stage.replace('_refinement', '').slice(1) + ' Refinement' : stage.charAt(0).toUpperCase() + stage.slice(1)}</h2>
          </div>

          {/* Refinement stage instruction message */}
          {stage === 'impression_refinement' && (
            <div style={{
              marginBottom: '20px',
              padding: '15px',
              backgroundColor: '#e7f3ff',
              borderRadius: '8px',
              border: '1px solid #b3d9ff'
            }}>
              <p style={{ margin: '0', color: '#004085', fontSize: '16px' }}>
                <strong>Refinement Stage:</strong> Review the 4 refined variations below. These images converge on your preferred concepts from the previous stage. Simply select your favorite image and continue.
              </p>
            </div>
          )}

          {/* Toggle button for tags (hidden for refinement stages) */}
          {stage !== 'impression_refinement' && (
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
                onMouseEnter={e => {
                  e.target.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={e => {
                  e.target.style.transform = 'scale(1)';
                }}
              >
                {showTagsByDefault ? 'Collapse Tags' : 'Expand Tags'}
              </button>
            </div>
          )}

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
                      {/* Hide Visual Tags button for refinement stages */}
                      {!stage === 'impression_refinement' && (
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
                          onMouseEnter={e => {
                            e.target.style.transform = 'scale(1.05)';
                            e.target.style.backgroundColor = '#f0f9ff';
                          }}
                          onMouseLeave={e => {
                            e.target.style.transform = 'scale(1)';
                            e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                          }}
                        >
                          Visual Tags
                        </button>
                      )}
                      
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
                        onMouseEnter={e => {
                          e.target.style.transform = 'scale(1.05)';
                          e.target.style.backgroundColor = '#f0f9ff';
                        }}
                        onMouseLeave={e => {
                          e.target.style.transform = 'scale(1)';
                          e.target.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                        }}
                      >
                        {stage === 'impression_refinement' ? 'Tag Weights' : 'JSON Script'}
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

                    {/* Inline tag display (hidden for refinement stages) */}
                    {showTagsByDefault && stage !== 'impression_refinement' && (
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
            
            {/* Right column: Bubble chart OR Refinement controls */}
            <div style={{ flex: '0 0 50%', minWidth: '0' }}>
              {stage === 'impression_refinement' ? (
                /* Refinement Iteration Controls */
                <RefinementIterationControls
                  sessionId={sessionId}
                  stage={stage.replace('_refinement', '')}  // Pass base stage
                  images={images}
                  selectedImage={selectedImage}
                  initialRound={refinementRound}
                  disabled={isLoading}
                  onContinue={handleContinue}  // Continue to next stage
                  onRefinementComplete={(newImages, round) => {
                    // Update images with new round
                    setImages(newImages);
                    setRefinementRound(round);
                    setSelectedImage(null);  // Reset selection for new round
                  }}
                />
              ) : (
                /* Bubble chart for exploration stage */
                <ConceptRefinementPanel
                  sessionId={sessionId}
                  stage={stage}
                  images={images}
                  selectedImage={selectedImage}
                  onImageSelect={handleSelect}
                  onTagClick={conceptTagHandlerRef}
                  onTagPreferencesUpdate={handleConceptTagPreferencesUpdate}
                />
              )}
            </div>
          </div>


          {/* Continue Button (for non-refinement stages) */}
          {stage !== 'impression_refinement' && (
            <button
              onClick={handleContinue}
              disabled={!selectedImage || isLoading}
              style={{
                marginTop: '20px',
                padding: '10px 20px',
                backgroundColor: isLoading ? '#ccc' : (selectedImage ? buttonColor : '#ccc'),
                color: 'white',
                border: 'none',
                borderRadius: '5px',
                cursor: (!selectedImage || isLoading) ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.3s ease'
              }}
            >
              {isLoading ? 'Processing...' : 'Continue to Next Stage'}
            </button>
          )}
        </div>
      )}
      
      {statusMessages.length > 0 && (
        <GenerationStatus messages={statusMessages} />
      )}

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