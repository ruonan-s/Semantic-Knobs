import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ProgressBar from './components/ProgressBar';
import NavigationBar from './components/NavigationBar';
import GenerationStatus from './components/GenerationStatus';
import TagSidebar from './components/TagSidebar';
import JsonPanel from './components/JsonPanel';
import InlineTagDisplay from './components/InlineTagDisplay';
import ConceptRefinementPanel from './components/ConceptRefinementPanel';
import RefinementIterationControls from './components/RefinementIterationControls';
import SessionBrowser from './components/SessionBrowser';
import StageSelector from './components/StageSelector';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [stage, setStage] = useState('landing');
  const [mode, setMode] = useState('sequential'); // 'sequential' or 'cumulative_tags'
  
  // Test Stage Refinement state
  const [testStageMode, setTestStageMode] = useState({
    active: false,
    sessionPath: null,
    sessionName: null,
    currentStage: null,
    descriptor: '',
    loadedImages: [],
    loadedTags: {},
    stageJson: [],
    selectedConceptIndex: 0
  });
  
  // Define refinement stages
  const REFINEMENT_STAGES = ['impression_refinement', 'spatial_refinement', 'objects_refinement', 'ambient_refinement'];
  const isRefinementStage = REFINEMENT_STAGES.includes(stage);
  
  // Refinement iteration state
  const [refinementRound, setRefinementRound] = useState(1);
  const [cumulativeTagsState, setCumulativeTagsState] = useState({
    conceptIndex: 0,
    totalConcepts: 4,
    hasNext: true,
    currentImageTags: []
  });
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [inputValue, setInputValue] = useState('');
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
  const [availableModes, setAvailableModes] = useState([]);
  const [finalModeResults, setFinalModeResults] = useState({});
  const [currentFinalMode, setCurrentFinalMode] = useState(null);
  const [uploadedFolder, setUploadedFolder] = useState(null);
  const [validationError, setValidationError] = useState(null);
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

  // Load tags for all images when images change (skip for refinement stages)
  useEffect(() => {
    if (images && images.length > 0 && sessionId && showTagsByDefault && !isRefinementStage) {
      loadAllImageTags(images);
    }
  }, [images, sessionId, showTagsByDefault, stage, isRefinementStage]);

  // Function to load JSON for an image
  const loadImageJson = async (imageId) => {
    try {
      // Determine the correct stage for the API call
      let apiStage = stage;
      if (stage === 'cumulative_tags') {
        // In cumulative tags mode, determine the actual stage from the image ID
        // Image ID format: impression_0_0_0, spatial_1_0_0, etc.
        const stageName = imageId.split('_')[0];
        apiStage = stageName;
      }

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
    setConceptTagPreferences(tagPrefs);
  }, []);

  // Function to handle tag preferences (now concept-based)
  const handleTagPreference = (tag, preference, imageId) => {
    // ===== DEBUG LOGS: Tag Click =====
    console.log('👆 [DEBUG] TAG CLICK (Concept-based):');
    console.log('  Tag:', tag);
    console.log('  Preference:', preference);
    console.log('  Image ID:', imageId);
    console.log('  Current Stage:', stage);
    
    // Determine the actual stage
    let actualStage = stage;
    if (stage === 'cumulative_tags') {
      actualStage = cumulativeTagsState.currentStage || 
                   (imageId ? imageId.split('_')[0] : 'impression');
    }
    
    // Build tag_id for concept system
    // Format: tag_{stage}_{imageId}_{index}
    // We need to find the index of this tag in the image's tags
    const imageTags = imageTagsMap[imageId] || [];
    const tagIndex = imageTags.findIndex(t => t === tag);
    
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
      return;
    }
    
    const tagId = `tag_${actualStage}_${imageId}_${tagIndex}`;
    console.log('  ✅ Tag ID:', tagId);
    
    // Call concept handler if available
    if (conceptTagHandlerRef.current) {
      conceptTagHandlerRef.current(tagId, preference);
      console.log('✅ [DEBUG] Concept handler called');
    } else {
      console.warn('⚠️  Concept handler not initialized yet');
    }
  };

  // Derive UI preferences from concept system for tag components
  const derivedTagPreferences = useMemo(() => {
    // Determine the actual stage
    let actualStage = stage;
    if (stage === 'cumulative_tags') {
      actualStage = cumulativeTagsState.currentStage || 'impression';
    }

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
  }, [conceptTagPreferences, imageTagsMap, stage, cumulativeTagsState.currentStage]);

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

  const handleSubmit = async (descriptor) => {
    if (mode === 'sequential') {
      await handleSequentialSubmit(descriptor);
    } else if (mode === 'cumulative_tags') {
      await handleCumulativeTagsSubmit(descriptor);
    }
  };

  const handleSequentialSubmit = async (descriptor) => {
    try {
      setIsLoading(true);
      addStatusMessage(`Starting fast sequential generation for: ${descriptor}`);
      
      const res = await fetch('/api/generate-fast', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ descriptor }),
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


  const handleCumulativeTagsSubmit = async (descriptor) => {
    try {
      setIsLoading(true);
      addStatusMessage(`Starting cumulative tags generation for: ${descriptor}`);
      
      const res = await fetch('/api/generate-cumulative-tags', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ descriptor }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      setSessionId(data.session_id);
      setStage('cumulative_tags');
      setImages(data.images);
      
      // Initialize cumulative tags state
      setCumulativeTagsState({
        conceptIndex: 0,
        totalConcepts: 4,
        hasNext: true,
        currentImageTags: [],
        currentStage: 'impression'
      });
      
      addStatusMessage(`Generated first concept for ${data.stage} stage`);
      
    } catch (error) {
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCumulativeTagsFeedback = async (positiveTagsInput, negativeTagsInput) => {
    try {
      setIsLoading(true);
      
      // Get the actual stage from the current image based on concept index
      const currentImage = images.find(img => {
        const imgConceptIndex = parseInt(img.id.split('_')[1]);
        return imgConceptIndex === cumulativeTagsState.conceptIndex;
      });
      const currentImageId = currentImage?.id;
      const actualStage = cumulativeTagsState.currentStage || 
                         (currentImageId ? currentImageId.split('_')[0] : 'impression');
      
      // ===== DEBUG LOGS: Frontend Tag Submission =====
      console.log('🏷️ [DEBUG] FRONTEND TAG FEEDBACK:');
      console.log('  Current Image ID:', currentImageId);
      console.log('  Actual Stage:', actualStage);
      console.log('  Session ID:', sessionId);
      console.log('  Concept Index:', cumulativeTagsState.conceptIndex);
      console.log('  Positive Tags:', positiveTagsInput);
      console.log('  Negative Tags:', negativeTagsInput);
      console.log('  User Preferences:', userPreferences);
      
      const res = await fetch('/api/cumulative-tags-feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          stage: actualStage,
          concept_index: cumulativeTagsState.conceptIndex,
          positive_tags: positiveTagsInput,
          negative_tags: negativeTagsInput
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.status === 'stage_complete') {
        // All concepts for this stage completed
        addStatusMessage('All concepts for this stage completed. You can select your preferred concept and continue to the next stage.');
        setCumulativeTagsState(prev => ({
          ...prev,
          hasNext: false
        }));
      } else {
        // Next concept generated - accumulate images instead of replacing
        setImages(prev => {
          const newImages = [...prev];
          // Add the new image(s) to the array
          data.images.forEach(newImage => {
            const existingIndex = newImages.findIndex(img => img.id === newImage.id);
            if (existingIndex >= 0) {
              newImages[existingIndex] = newImage;
            } else {
              newImages.push(newImage);
            }
          });
          return newImages;
        });
        setCumulativeTagsState(prev => ({
          ...prev,
          conceptIndex: data.concept_index,
          hasNext: data.has_next,
          currentImageTags: []
        }));
        addStatusMessage(`Generated concept ${data.concept_index + 1} of ${data.total_concepts}`);
      }
      
    } catch (error) {
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCumulativeTagsNextStage = async (selectedConcept) => {
    try {
      setIsLoading(true);
      
      const res = await fetch('/api/cumulative-tags-next-stage', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId,
          selected_concept: selectedConcept
        }),
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      if (data.status === 'complete') {
        if (data.next_stage === 'mode-selection') {
          // All stages completed, transition to mode selection
          addStatusMessage('All stages completed! Choose your final generation mode.');
          setStage('mode-selection');
          setImages([]);
          setSelectedImage(null);
        } else {
          // Just completed, show final stage
          addStatusMessage('All stages completed!');
          setStage('final');
        }
      } else {
        // Update state for next stage - keep stage as 'cumulative_tags'
        // but track the actual stage (spatial, objects, etc.) in cumulative state
        setImages(data.images);
        setCumulativeTagsState({
          conceptIndex: 0,
          totalConcepts: 4,
          hasNext: true,
          currentImageTags: [],
          currentStage: data.stage  // Track the actual stage here
        });
        
        addStatusMessage(`Moved to ${data.stage} stage`);
      }
      
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

  const handleContinue = async () => {
    try {
      setIsLoading(true);
      setButtonColor('#4CAF50'); // Change to green when continuing
      
      if (mode === 'cumulative_tags') {
        // Handle cumulative tags next stage
        await handleCumulativeTagsNextStage(selectedImage);
      } else {
        // Handle regular sequential/parallel mode
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            session_id: sessionId,
            stage,
            selected_image_id: selectedImage,
            preferences: userPreferences // Send the complete preferences object
          }),
        });
        
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        
        const data = await res.json();
        if (data.next_stage) {
          setStage(data.next_stage);
          if (data.next_stage === 'mode-selection') {
            setImages([]);
          } else {
            setImages(data.images);
          }
          setSelectedImage(null);
          setButtonColor('#007bff'); // Reset color for next stage
          setShowTagDrawer(false); // Close sidebar when moving to next stage
          setShowJsonPanel(false); // Close JSON panel when moving to next stage
          
          // Reset refinement round when entering a refinement stage
          if (REFINEMENT_STAGES.includes(data.next_stage)) {
            setRefinementRound(1);
          }
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
    if (inputValue.trim()) {
      handleSubmit(inputValue.trim());
      setInputValue('');
    }
  };


  // Cleanup polling intervals when component unmounts or stage changes
  useEffect(() => {
    return () => {
      if (statusPollingRef.current?.statusPolling) {
        clearInterval(statusPollingRef.current.statusPolling);
      }
    };
  }, []);

  // Fetch available modes when entering mode-selection stage
  const fetchAvailableModes = async () => {
    try {
      const res = await fetch(`/api/mode-selection/${sessionId}`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setAvailableModes(data.modes);
    } catch (error) {
      addStatusMessage(`Error fetching modes: ${error.message}`);
    }
  };

  // Handle mode selection and final generation
  const handleModeSelect = async (modeId) => {
    try {
      setIsLoading(true);
      setCurrentFinalMode(modeId);
      addStatusMessage(`Generating final images using ${modeId}...`);
      
      // Check if this is a cumulative tags session
      const isCumulativeTags = sessionId && sessionId.startsWith('[cumtag]');
      
      let res;
      if (isCumulativeTags) {
        // Use cumulative tags final endpoint
        res = await fetch(`/api/generate-final-cumulative-tags/${modeId}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            session_id: sessionId,
            stage: 'final',
            selected_image_id: null,
            preferences: userPreferences
          }),
        });
      } else {
        // Use regular final endpoint
        res = await fetch(`/api/generate-final/${modeId}`, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            session_id: sessionId,
            stage: 'final',
            selected_image_id: null,
            preferences: userPreferences
          }),
        });
      }
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      
      // Store results for this mode
      setFinalModeResults(prev => ({
        ...prev,
        [modeId]: {
          images: data.images,
          mode_name: data.mode_name
        }
      }));
      
      // Show the final results
      setStage('final');
      setImages(data.images);
      setSelectedImage(null);
      
      addStatusMessage(`Final stage completed using ${data.mode_name}!`);
      
      // Start polling for status messages
      startStatusPolling(sessionId);
    } catch (error) {
      addStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle going back to mode selection
  const handleBackToModeSelection = () => {
    setStage('mode-selection');
    setImages([]);
    setSelectedImage(null);
    setCurrentFinalMode(null);
  };

  // Fetch modes when stage becomes mode-selection
  useEffect(() => {
    if (stage === 'mode-selection' && sessionId) {
      fetchAvailableModes();
    }
  }, [stage, sessionId]);

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
    // Validate required structure
    const requiredFiles = [
      'impression/impression.json',
      'spatial/spatial.json', 
      'ambient/ambient.json',
      'preferences.json'
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
      
      // Set session data and jump to mode selection
      setSessionId(data.session_id);
      setUserPreferences(data.preferences);
      setStage('mode-selection');
      
      addStatusMessage('Session loaded successfully! Choose a final generation mode.');
      
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
                🎯 Test Final Stage
              </h3>
              <p style={{ margin: '0', color: '#333', fontSize: '14px', lineHeight: '1.5' }}>
                Upload session folder and test final generation modes
              </p>
            </div>

            {/* NEW: Test Stage Refinement Button */}
            <div
              onClick={() => setStage('test-stage-refinement')}
              style={{
                border: '2px solid #ff9800',
                borderRadius: '12px',
                padding: '30px 20px',
                backgroundColor: '#fff8e1',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
              }}
              onMouseEnter={e => {
                e.currentTarget.style.backgroundColor = '#fff3cd';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.backgroundColor = '#fff8e1';
                e.currentTarget.style.transform = 'translateY(0px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
              }}
            >
              <h3 style={{ margin: '0 0 12px 0', color: '#ff9800', fontSize: '20px' }}>
                ⚡ Test Stage Refinement
              </h3>
              <p style={{ margin: '0', color: '#333', fontSize: '14px', lineHeight: '1.5' }}>
                Resume from any stage, refine with concepts, continue pipeline
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
            <h2 style={{ margin: '0', color: '#333' }}>Upload Session Folder</h2>
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
      ) : stage === 'test-stage-refinement' ? (
        // Test Stage Refinement: Session Browser
        <SessionBrowser 
          onSessionSelect={(sessionPath) => {
            console.log('[TEST STAGE] Session selected:', sessionPath);
            setTestStageMode({
              ...testStageMode,
              active: true,
              sessionPath: sessionPath,
              sessionName: sessionPath
            });
            setStage('test-stage-select-stage');
          }}
        />
      ) : stage === 'test-stage-select-stage' ? (
        // Test Stage Refinement: Stage Selector
        <StageSelector 
          sessionPath={testStageMode.sessionPath}
          sessionName={testStageMode.sessionName}
          onStageSelect={async (stageName) => {
            console.log('[TEST STAGE] Stage selected:', stageName);
            
            try {
              setIsLoading(true);
              addStatusMessage(`Loading ${stageName} data from session...`);
              
              // Load stage data
              const response = await fetch('/api/load-stage-data', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  session_path: testStageMode.sessionPath,
                  stage: stageName
                })
              });
              
              if (!response.ok) {
                throw new Error(`Failed to load stage data: ${response.status}`);
              }
              
              const data = await response.json();
              
              console.log('[TEST STAGE] Stage data loaded:', {
                images: data.images.length,
                tagKeys: Object.keys(data.tags).length,
                stageJson: data.stage_json.length
              });
              
              // Extract descriptor from stage JSON if available
              let descriptor = '';
              if (data.stage_json && data.stage_json.length > 0) {
                descriptor = data.stage_json[0].user_description || '';
              }
              
              // Set up test stage mode
              setTestStageMode({
                ...testStageMode,
                currentStage: stageName,
                loadedImages: data.images,
                loadedTags: data.tags,
                stageJson: data.stage_json,
                descriptor: descriptor,
                selectedConceptIndex: 0
              });
              
              // Set images and stage
              setImages(data.images);
              setImageTagsMap(data.tags);
              setSessionId(testStageMode.sessionPath); // Use session path as ID
              setStage(stageName); // Move to the stage view with concept refinement
              
              addStatusMessage(`Loaded ${data.images.length} images with tags`);
              addStatusMessage('Ready to refine concepts - like/dislike tags or drag to reorder');
              
            } catch (error) {
              console.error('[TEST STAGE] Error loading stage:', error);
              addStatusMessage(`Error: ${error.message}`);
            } finally {
              setIsLoading(false);
            }
          }}
          onBack={() => {
            setStage('test-stage-refinement');
            setTestStageMode({
              active: false,
              sessionPath: null,
              sessionName: null,
              currentStage: null,
              descriptor: '',
              loadedImages: [],
              loadedTags: {},
              stageJson: [],
              selectedConceptIndex: 0
            });
          }}
        />
      ) : stage === 'input' ? (
        <div>
          <h2>Enter Design Description</h2>
          
          {/* Mode Selection */}
          <div style={{ marginBottom: '20px' }}>
            <h3 style={{ marginBottom: '10px', fontSize: '16px' }}>Generation Mode:</h3>
            <div style={{ display: 'flex', gap: '15px' }}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="radio"
                  value="sequential"
                  checked={mode === 'sequential'}
                  onChange={(e) => setMode(e.target.value)}
                  style={{ marginRight: '8px' }}
                />
                <span>Sequential (Impression → Spatial → Objects → Ambient → Final)</span>
              </label>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <input
                  type="radio"
                  value="cumulative_tags"
                  checked={mode === 'cumulative_tags'}
                  onChange={(e) => setMode(e.target.value)}
                  style={{ marginRight: '8px' }}
                />
                <span>Cumulative Tags (Concept-by-concept with feedback)</span>
              </label>
            </div>
          </div>
          
          <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleInputSubmit();
                }
              }}
              placeholder="Describe your desired environment..."
              style={{
                flex: 1,
                padding: '10px',
                fontSize: '16px',
                borderRadius: '4px',
                border: '1px solid #ccc'
              }}
              disabled={isLoading}
            />
            <button
              onClick={handleInputSubmit}
              disabled={!inputValue.trim() || isLoading}
              style={{
                padding: '10px 20px',
                fontSize: '16px',
                backgroundColor: isLoading ? '#ccc' : (inputValue.trim() ? buttonColor : '#ccc'),
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: (!inputValue.trim() || isLoading) ? 'not-allowed' : 'pointer',
                transition: 'background-color 0.3s ease'
              }}
            >
              {isLoading ? 'Generating...' : 'Send'}
            </button>
          </div>
        </div>
      ) : stage === 'mode-selection' ? (
        <div>
          <h2>Choose Final Generation Mode</h2>
          <p style={{ marginBottom: '30px', color: '#666', fontSize: '16px' }}>
            Select how you want to generate your final environment images:
          </p>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '30px' }}>
            {availableModes.map((mode) => (
              <div
                key={mode.id}
                style={{
                  border: '2px solid #e0e0e0',
                  borderRadius: '12px',
                  padding: '25px',
                  backgroundColor: '#f9f9f9',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}
                onMouseEnter={e => {
                  e.target.style.borderColor = '#007bff';
                  e.target.style.backgroundColor = '#f0f8ff';
                  e.target.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={e => {
                  e.target.style.borderColor = '#e0e0e0';
                  e.target.style.backgroundColor = '#f9f9f9';
                  e.target.style.transform = 'translateY(0px)';
                }}
                onClick={() => handleModeSelect(mode.id)}
              >
                <h3 style={{ 
                  margin: '0 0 15px 0', 
                  color: '#333',
                  fontSize: '20px',
                  fontWeight: '600'
                }}>
                  {mode.name}
                </h3>
                <p style={{ 
                  margin: '0', 
                  color: '#666',
                  fontSize: '14px',
                  lineHeight: '1.5'
                }}>
                  {mode.description}
                </p>
                
                {/* Show if this mode has been used before */}
                {finalModeResults[mode.id] && (
                  <div style={{
                    marginTop: '15px',
                    padding: '10px',
                    backgroundColor: '#e8f5e8',
                    borderRadius: '6px',
                    fontSize: '12px',
                    color: '#2d5a2d'
                  }}>
                    ✓ Already generated {finalModeResults[mode.id].images?.length || 0} images
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {isLoading && (
            <div style={{ textAlign: 'center', color: '#666', fontSize: '16px' }}>
              <p>Generating final images... This may take a few moments.</p>
            </div>
          )}
          
          {Object.keys(finalModeResults).length > 0 && (
            <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f0f8ff', borderRadius: '8px' }}>
              <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>Previous Results</h3>
              <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
                {Object.entries(finalModeResults).map(([modeId, result]) => (
                  <button
                    key={modeId}
                    onClick={() => {
                      setStage('final');
                      setImages(result.images);
                      setCurrentFinalMode(modeId);
                    }}
                    style={{
                      padding: '8px 16px',
                      backgroundColor: '#007bff',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      fontSize: '14px'
                    }}
                  >
                    View {result.mode_name} ({result.images?.length || 0} images)
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : stage === 'cumulative_tags' ? (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2>Cumulative Tags Mode: {cumulativeTagsState.currentStage || 'impression'}</h2>
            <div style={{ fontSize: '14px', color: '#666' }}>
              Concept {cumulativeTagsState.conceptIndex + 1} of {cumulativeTagsState.totalConcepts}
              {!cumulativeTagsState.hasNext && ' - Review and provide feedback on visual tags'}
            </div>
          </div>
          
          <div style={{ marginBottom: '20px', textAlign: 'center' }}>
            <p style={{ color: '#666', fontSize: '16px' }}>
              {cumulativeTagsState.hasNext 
                ? 'Review the concepts and provide feedback on visual tags to improve the next concept.' 
                : 'All concepts generated for this stage. Select your preferred image and continue to the next stage.'
              }
            </p>
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
            <span style={{ fontWeight: '500', color: '#495057' }}>
              Tags Display: {showTagsByDefault ? 'Expanded' : 'Collapsed'}
            </span>
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

          <div style={{ display: 'flex', gap: '20px', overflowX: 'auto', marginBottom: '20px', minHeight: '300px' }}>
            {images
              .sort((a, b) => {
                // Sort by concept index (impression_0_0_0, impression_1_0_0, etc.)
                const aIndex = parseInt(a.id.split('_')[1]);
                const bIndex = parseInt(b.id.split('_')[1]);
                return aIndex - bIndex;
              })
              .map((image, index) => (
              <div
                key={image.id}
                style={{
                  position: 'relative',
                  border: selectedImage === image.id ? '3px solid blue' : 
                         parseInt(image.id.split('_')[1]) === cumulativeTagsState.conceptIndex ? '2px solid orange' :
                         '1px solid gray',
                  padding: '10px',
                  borderRadius: '4px',
                  transition: 'all 0.3s ease',
                  minWidth: '250px',
                  maxWidth: '300px',
                  flexShrink: 0,
                  backgroundColor: parseInt(image.id.split('_')[1]) === cumulativeTagsState.conceptIndex ? 
                                  'rgba(255, 165, 0, 0.1)' : 'transparent',
                  display: 'flex',
                  flexDirection: 'column'
                }}
                ref={el => imageRefs.current[image.id] = el}
              >
                {/* Concept label */}
                <div style={{
                  position: 'absolute',
                  top: '5px',
                  left: '5px',
                  backgroundColor: 'rgba(0, 0, 0, 0.7)',
                  color: 'white',
                  padding: '4px 8px',
                  borderRadius: '4px',
                  fontSize: '12px',
                  fontWeight: 'bold',
                  zIndex: 10
                }}>
                  Concept {parseInt(image.id.split('_')[1]) + 1}
                </div>
                
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
                    tags={imageTagsMap[image.id] || []}
                    imageId={image.id}
                    onTagPreference={handleTagPreference}
                    preferences={derivedTagPreferences}
                  />
                )}
              </div>
            ))}
          </div>
          
          {/* Concept Refinement Panel for Cumulative Tags */}
          <ConceptRefinementPanel
            sessionId={sessionId}
            stage={cumulativeTagsState.currentStage || 'impression'}
            images={images}
            selectedImage={selectedImage}
            onImageSelect={handleSelect}
            onTagClick={conceptTagHandlerRef}
            onTagPreferencesUpdate={handleConceptTagPreferencesUpdate}
          />
          
          <button
            onClick={() => {
              console.log('🔘 [DEBUG] BUTTON CLICKED - Generate Next Concept');
              if (cumulativeTagsState.hasNext) {
                // Process tag feedback - get tags from user preferences for current stage and image
                // Find the current image based on concept index
                const currentImage = images.find(img => {
                  const imgConceptIndex = parseInt(img.id.split('_')[1]);
                  return imgConceptIndex === cumulativeTagsState.conceptIndex;
                });
                const currentImageId = currentImage?.id;
                const actualStage = cumulativeTagsState.currentStage || 
                                   (currentImageId ? currentImageId.split('_')[0] : 'impression');
                const stageTagPrefs = userPreferences.tags[actualStage] || [];
                
                // Filter tags for the current image and collect positive/negative
                const currentImageTags = stageTagPrefs.filter(t => t.source_image === currentImageId);
                const positiveTagsFromSidebar = currentImageTags.filter(t => t.preference === 'positive').map(t => t.tag);
                const negativeTagsFromSidebar = currentImageTags.filter(t => t.preference === 'negative').map(t => t.tag);
                
                // ===== DEBUG LOGS: Button Click Tag Collection =====
                console.log('🔘 [DEBUG] BUTTON CLICK - TAG COLLECTION:');
                console.log('  Current Image ID:', currentImageId);
                console.log('  Actual Stage:', actualStage);
                console.log('  Stage Tag Preferences:', stageTagPrefs);
                console.log('  Current Image Tags:', currentImageTags);
                console.log('  Positive Tags Collected:', positiveTagsFromSidebar);
                console.log('  Negative Tags Collected:', negativeTagsFromSidebar);
                console.log('  All User Preferences:', userPreferences);
                
                handleCumulativeTagsFeedback(positiveTagsFromSidebar, negativeTagsFromSidebar);
              } else {
                // Continue to next stage - requires image selection
                handleContinue();
              }
            }}
            disabled={(!cumulativeTagsState.hasNext && !selectedImage) || isLoading}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              backgroundColor: isLoading ? '#ccc' : 
                            (cumulativeTagsState.hasNext ? buttonColor : (selectedImage ? buttonColor : '#ccc')),
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: ((!cumulativeTagsState.hasNext && !selectedImage) || isLoading) ? 'not-allowed' : 'pointer',
              transition: 'background-color 0.3s ease'
            }}
          >
            {isLoading ? 'Processing...' : 
             cumulativeTagsState.hasNext ? 'Generate Next Concept' : 'Continue to Next Stage'}
          </button>
        </div>
        ) : (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h2>Current Stage: {stage.includes('_refinement') ? stage.replace('_refinement', '').charAt(0).toUpperCase() + stage.replace('_refinement', '').slice(1) + ' Refinement' : stage.charAt(0).toUpperCase() + stage.slice(1)}</h2>
            {stage === 'final' && (
              <button
                onClick={handleBackToModeSelection}
                style={{
                  padding: '10px 20px',
                  backgroundColor: '#6c757d',
                  color: 'white',
                  border: 'none',
                  borderRadius: '5px',
                  cursor: 'pointer',
                  transition: 'background-color 0.3s ease'
                }}
                onMouseEnter={e => e.target.style.backgroundColor = '#5a6268'}
                onMouseLeave={e => e.target.style.backgroundColor = '#6c757d'}
              >
                ← Try Different Mode
              </button>
            )}
          </div>
          
          {/* Refinement stage instruction message */}
          {isRefinementStage && (
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
          
          {stage === 'final' && currentFinalMode && (
            <div style={{ 
              marginBottom: '20px', 
              padding: '15px', 
              backgroundColor: '#e8f5e8', 
              borderRadius: '8px',
              border: '1px solid #d4edda'
            }}>
              <p style={{ margin: '0', color: '#2d5a2d', fontSize: '14px' }}>
                <strong>Generated using:</strong> {finalModeResults[currentFinalMode]?.mode_name || currentFinalMode}
              </p>
            </div>
          )}
          
          {/* Toggle button for tags (hidden for refinement stages) */}
          {!isRefinementStage && (
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
              <span style={{ fontWeight: '500', color: '#495057' }}>
                Tags Display: {showTagsByDefault ? 'Expanded' : 'Collapsed'}
              </span>
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

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '20px' }}>
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
                  flexDirection: 'column'
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
                  {!isRefinementStage && (
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

                {/* Inline tag display (hidden for refinement stages) */}
                {showTagsByDefault && !isRefinementStage && (
                  <InlineTagDisplay
                    tags={imageTagsMap[image.id] || []}
                    imageId={image.id}
                    onTagPreference={handleTagPreference}
                    preferences={derivedTagPreferences}
                  />
                )}
              </div>
            ))}
          </div>
          
          {/* Concept Refinement Panel (hidden for refinement stages) */}
          {stage !== 'final' && stage !== 'mode-selection' && !isRefinementStage && (
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
          
          {/* Test Stage Refinement: Generate Refinement Button */}
          {testStageMode.active && testStageMode.currentStage && (
            <button
              onClick={async () => {
                try {
                  setIsLoading(true);
                  addStatusMessage('Generating refinement images with your concept preferences...');
                  
                  // Get categorized concepts from the refinement panel state
                  // For now, we'll use the conceptTagPreferences to derive positive/negative labels
                  const positiveConcepts = [];
                  const negativeConcepts = [];
                  
                  // Extract concept labels from tag preferences
                  // This is a simplified approach - in production you'd get this from ConceptRefinementPanel
                  for (const [tagId, preference] of Object.entries(conceptTagPreferences)) {
                    if (preference === 'positive') {
                      // Extract tag text from imageTagsMap
                      const parts = tagId.split('_');
                      const tagIndex = parseInt(parts[parts.length - 1]);
                      const imageId = parts.slice(2, parts.length - 1).join('_');
                      const imageTags = imageTagsMap[imageId] || [];
                      const tagText = imageTags[tagIndex];
                      if (tagText && !positiveConcepts.includes(tagText)) {
                        positiveConcepts.push(tagText);
                      }
                    } else if (preference === 'negative') {
                      const parts = tagId.split('_');
                      const tagIndex = parseInt(parts[parts.length - 1]);
                      const imageId = parts.slice(2, parts.length - 1).join('_');
                      const imageTags = imageTagsMap[imageId] || [];
                      const tagText = imageTags[tagIndex];
                      if (tagText && !negativeConcepts.includes(tagText)) {
                        negativeConcepts.push(tagText);
                      }
                    }
                  }
                  
                  console.log('[GENERATE REFINEMENT] Calling API with:', {
                    sessionPath: testStageMode.sessionPath,
                    stage: testStageMode.currentStage,
                    selectedConceptIndex: testStageMode.selectedConceptIndex,
                    positiveConcepts,
                    negativeConcepts,
                    descriptor: testStageMode.descriptor
                  });
                  
                  const response = await fetch('/api/generate-stage-refinement', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      session_path: testStageMode.sessionPath,
                      stage: testStageMode.currentStage,
                      selected_concept_index: testStageMode.selectedConceptIndex,
                      positive_concept_labels: positiveConcepts,
                      negative_concept_labels: negativeConcepts,
                      descriptor: testStageMode.descriptor
                    })
                  });
                  
                  if (!response.ok) {
                    throw new Error(`Refinement generation failed: ${response.status}`);
                  }
                  
                  const data = await response.json();
                  
                  console.log('[GENERATE REFINEMENT] Success:', {
                    imageCount: data.images.length,
                    refinementFolder: data.refinement_folder
                  });
                  
                  // Display the new refinement images
                  setImages(data.images);
                  addStatusMessage(`✅ Refinement complete! Generated ${data.images.length} new images`);
                  addStatusMessage(`Saved to: ${data.refinement_folder}/`);
                  
                  // Update stage to show refinement results
                  setStage(`${testStageMode.currentStage}_refinement`);
                  
                } catch (error) {
                  console.error('[GENERATE REFINEMENT] Error:', error);
                  addStatusMessage(`❌ Error: ${error.message}`);
                } finally {
                  setIsLoading(false);
                }
              }}
              disabled={isLoading}
              style={{
                position: 'fixed',
                bottom: '30px',
                right: '30px',
                padding: '16px 32px',
                backgroundColor: isLoading ? '#ccc' : '#ff9800',
                color: 'white',
                border: 'none',
                borderRadius: '30px',
                fontSize: '16px',
                fontWeight: 'bold',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 12px rgba(255,152,0,0.4)',
                zIndex: 1000,
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.3s ease'
              }}
              onMouseEnter={(e) => {
                if (!isLoading) {
                  e.target.style.backgroundColor = '#f57c00';
                  e.target.style.boxShadow = '0 6px 20px rgba(255,152,0,0.6)';
                  e.target.style.transform = 'scale(1.05)';
                }
              }}
              onMouseLeave={(e) => {
                if (!isLoading) {
                  e.target.style.backgroundColor = '#ff9800';
                  e.target.style.boxShadow = '0 4px 12px rgba(255,152,0,0.4)';
                  e.target.style.transform = 'scale(1)';
                }
              }}
            >
              {isLoading ? (
                <>
                  <span>⏳</span>
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <span>⚡</span>
                  <span>Generate Refinement</span>
                </>
              )}
            </button>
          )}
          
          {/* Refinement Iteration Controls or Continue Button */}
          {isRefinementStage ? (
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