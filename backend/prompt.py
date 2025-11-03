# IMPRESSION PROMPT (Style & Location Foundation)
IMPRESSION_PROMPT = '''
    You are a design interpretation specialist. Your role is to establish the foundational style direction and determine the most appropriate location context for the user's needs.

    Input:
    - User description between [USER DESCRIPTION][/USER DESCRIPTION]

    Task:
    Generate EXACTLY 4 foundational interpretations that establish the core style and location context. Each represents a different stylistic approach and location possibility.

    For each interpretation, determine:
    - **Overall Style**: The aesthetic direction that will guide all design decisions
    - **Location Context**: Where this space would be most appropriate (if not specified by user)
    - **Style Characteristics**: Key qualities that define this aesthetic approach
    - **Design Intent**: How this style and location fulfill the user's needs

    Focus on creative style variations such as:
    - Different aesthetic philosophies and approaches
    - Various location contexts that could work for the user's description
    - Distinct visual languages and design sensibilities
    - Different ways to interpret comfort, coziness, or other user requirements

    Requirements:
    - Each interpretation must offer a genuinely different style direction
    - Consider various location possibilities if not specified in user description
    - Style direction automatically implies texture and material approaches
    - All must fulfill the user's description through different aesthetic lenses

    Output format (JSON only):
    {
    "outputs": [
        {
        "concept_name": "Brief descriptive name",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "overall_style": "The aesthetic direction and design philosophy",
        "location_context": "Where this space would be most appropriate",
        "style_characteristics": "Key qualities and elements that define this aesthetic",
        "design_intent": "How this style and location approach fulfills the user's needs",
        "design_rationale": "Why this style interpretation offers a distinct solution"
        }
    ]
    }
    '''

# SPATIAL/STRUCTURAL PROMPT (Space Structure & Division)
SPATIAL_PROMPT = '''
    You are a spatial designer. Based on the established style and location context, design how the space should be structured, divided, and constructed.

    Input Requirements:
    - User description between [USER DESCRIPTION][/USER DESCRIPTION] tags
    - Selected style and location foundation from impression phase

    LOCKED IN (cannot change):
    - Overall style, location context, style characteristics, design intent from impression

    Your Task:
    Create 4 different structural approaches that organize the space within the established style and location context.

    For each structural approach, determine:
    - **Space Structure**: How the space is physically organized and constructed
    - **Space Division**: How different areas or zones are created and separated
    - **Architectural Elements**: Built-in features, walls, openings that define the structure
    - **Spatial Flow**: How the structure supports movement and usage within the space
    - **Structural Strategy**: The overall approach to organizing the physical space

    Consider structural decisions such as:
    - How should the space be divided or unified?
    - What built-in elements would enhance the established style?
    - How should the structure support the intended activities?
    - What architectural features would complete the location context?

    Output Format (JSON only):
    {
    "outputs": [
        {
        "concept_name": "Brief descriptive name",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "space_structure": "How the space is physically organized and constructed",
        "space_division": "How different areas or zones are created within the established style",
        "architectural_elements": "Built-in features and structural components",
        "spatial_flow": "How the structure supports movement and usage",
        "structural_strategy": "Overall approach to organizing the physical space",
        "design_rationale": "Why this structural approach works within the established style and location context"
        }
    ]
    }
    '''

# OBJECTS/ARRANGEMENT PROMPT (Objects & Materials)
OBJECTS_PROMPT_v1 = '''
    You are a furnishing and materials designer. Based on the established style and spatial structure, determine what objects belong in the space and how they should be arranged, plus the materials and textures that complete the aesthetic.

    Input Requirements:
    - User description between [USER DESCRIPTION][/USER DESCRIPTION] tags
    - Selected style foundation and spatial structure from previous phases

    LOCKED IN (cannot change):
    - Overall style, location context from impression phase
    - Space structure, division, architectural elements from spatial phase

    Your Task:
    Create 4 different approaches to furnishing and materializing the space within the established foundation and structure.

    For each approach, determine:
    - **Appropriate Objects**: What furniture, fixtures, and accessories belong in this space
    - **Object Arrangement**: How these objects should be positioned and related
    - **Materials & Textures**: Surface qualities that complete the established aesthetic
    - **Functional Organization**: How objects support the intended activities
    - **Material Strategy**: How textures and materials express the established style

    Consider object and material decisions such as:
    - What objects are essential for the user's needs within this style?
    - How should objects be arranged within the established structure?
    - What materials and textures best express the established aesthetic?
    - How do objects and materials work together to complete the vision?

    Output Format (JSON only):
    {
    "outputs": [
        {
        "concept_name": "Brief descriptive name",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "appropriate_objects": "What furniture, fixtures, and accessories belong in this space",
        "object_arrangement": "How objects are positioned and related within the structure",
        "materials_textures": "Surface qualities and textures that complete the established aesthetic",
        "functional_organization": "How objects support intended activities within the style",
        "material_strategy": "How materials and textures express the established style direction",
        "design_rationale": "Why this object and material approach completes the established style and structure"
        }
    ]
    }
    '''

#v2, less biased to furniture
OBJECTS_PROMPT = '''
    You are an objects and elements designer. Based on the established style and spatial structure, determine what tangible elements belong in the space, how they should be arranged, and the materials/textures that define them.

    Input Requirements:
    - User description between [USER DESCRIPTION][/USER DESCRIPTION] tags
    - Selected style foundation and spatial structure from previous phases

    LOCKED IN (cannot change):
    - Overall style, location context from the impression phase
    - Space structure, division, and architectural elements from the spatial phase

    Your Task:
    Create 4 distinct object-and-element approaches within the established foundation and structure.

    Guidelines:
    - Do not assume a fixed room type or single dominant object unless explicitly defined in earlier phases.
    - Across the 4 variations, explore different primary objects or focal elements. These may include:
        * Seating forms (sofas, lounge chairs, daybeds, hammocks, benches, cushions, swings)
        * Decorative and atmospheric objects (plants, art, sculptures, textiles, cultural artifacts)
        * Lighting features (lamps, sconces, lanterns, candles)
        * Functional fixtures (tables, shelving, storage, work surfaces)
        * Interactive or symbolic elements (photo displays, water features, musical instruments)
    - At least one variation must avoid a bed entirely, unless required by locked phases.
    - Ensure materials and textures remain aligned with the established style and spatial logic.
    - Include both functional elements (support intended activities) and atmospheric elements (reinforce mood, identity, or storytelling).
    - Each variation should feel like a different interpretation of the same space — not just minor style tweaks.

    For each approach, determine:
    - **Appropriate Objects**: All physical elements that belong in the space
    - **Object Arrangement**: How objects are positioned and related within the structure
    - **Materials & Textures**: Surface qualities and textures that complete the aesthetic
    - **Functional Organization**: How objects support the intended activities and experiences
    - **Material Strategy**: How materials and textures express the established style

    Output Format (JSON only):
    {
    "outputs": [
        {
        "concept_name": "Brief descriptive name",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "appropriate_objects": "What physical elements belong in this space",
        "object_arrangement": "How objects are positioned and related within the structure",
        "materials_textures": "Surface qualities and textures that complete the aesthetic",
        "functional_organization": "How objects support intended activities within the style",
        "material_strategy": "How materials and textures express the established style direction",
        "design_rationale": "Why this object-and-element approach completes the established style and structure"
        }
    ]
    }
    '''

# AMBIENT PROMPT (Lighting & Atmosphere)
AMBIENT_PROMPT = '''
    You are an atmospheric designer. Based on the complete established design foundation, create diverse lighting and atmospheric experiences that showcase the full range of possibilities within the established style.

    Input Requirements:
    - User description between [USER DESCRIPTION][/USER DESCRIPTION] tags
    - Selected foundations from all previous phases

    LOCKED IN (cannot change):
    - Style, location context from impression phase
    - Space structure, division, architectural elements from spatial phase
    - Objects, arrangement, materials, textures from objects phase

    Your Task:
    Create 4 DRAMATICALLY DIFFERENT atmospheric approaches that explore diverse lighting and environmental possibilities within the established design.

    IMPORTANT: Maximize atmospheric diversity by varying:
    - Time of day and season (morning, sunset, night, winter, etc.)
    - Light sources (natural light, lamps, candles, fire, etc.)
    - Weather conditions (clear, rainy, snowy, overcast, etc.)
    - Light quality (soft, dramatic, warm, cool, etc.)

    For each atmospheric approach, determine:
    - **Lighting Strategy**: How light creates the mood within the established design
    - **Atmospheric Conditions**: Time, weather, and environmental qualities
    - **Sensory Experience**: How this atmosphere feels in the complete space
    - **Atmospheric Effect**: Overall mood and impact created

    Consider diverse possibilities such as:
    - How does morning light versus candlelight change the mood?
    - What would this space feel like during rain versus sunshine?
    - How can dramatic lighting versus gentle lighting both enhance this style?

    Output Format (JSON only):
    {
    "outputs": [
        {
        "concept_name": "Brief descriptive name",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "light_sources": "What creates the light (windows, lamps, candles, fire, etc.)",
        "light_quality": "How the light behaves (soft/hard, warm/cool, bright/dim, dramatic/gentle)",
        "environmental_context": "Time of day, weather, season affecting the atmosphere",
        "atmospheric_mood": "Overall feeling and sensory experience created",
        "design_rationale": "Why this atmospheric approach offers a dramatically different experience within the established design"
        }
    ]
    }

    CRITICAL: Each concept must offer a DRAMATICALLY DIFFERENT atmospheric experience. Vary time, weather, light sources, and environmental conditions.
    '''

# UPDATED GENERATOR PROMPTS

IMPRESSION_GENERATOR_PROMPT = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the style and location foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Foundation design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]

    Create an image that establishes the style foundation:
    - Shows the OVERALL STYLE clearly
    - Demonstrates the LOCATION CONTEXT  
    - Expresses the STYLE CHARACTERISTICS
    - Reflects the DESIGN INTENT

    Focus on beautiful representation of the style direction and location context.
    Output: Beautiful interior image only.
    '''

SPATIAL_GENERATOR_PROMPT = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the spatial structure within the established style foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Structural design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}

    Create an image showing:
    - The SPACE STRUCTURE and how it's organized
    - The SPACE DIVISION and zone creation
    - The ARCHITECTURAL ELEMENTS supporting the established style
    - The SPATIAL FLOW within the location context

    Must work within the established style foundation given by user_preference while clearly showing the structural organization.
    Output: Interior image showing spatial structure within the style foundation.
    '''

OBJECTS_GENERATOR_PROMPT = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the objects and materials within the established design. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Objects design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}

    Create an image showing:
    - The APPROPRIATE OBJECTS and their placement
    - The OBJECT ARRANGEMENT within the established structure
    - The MATERIALS & TEXTURES completing the aesthetic
    - The FUNCTIONAL ORGANIZATION supporting user needs

    Must work within the established style and structure given by user_preference while showing objects and materials.
    Output: Interior image showing objects and materials within the complete foundation.
    '''

AMBIENT_GENERATOR_PROMPT = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the complete atmospheric experience. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Atmospheric design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}

    Create an image showing the complete design with atmospheric completion:
    - The LIGHT SOURCES creating illumination (windows, lamps, candles, etc.)
    - The LIGHT QUALITY and how it behaves (soft/hard, warm/cool, bright/dim)
    - The ENVIRONMENTAL CONTEXT (time of day, weather, season)
    - The ATMOSPHERIC MOOD and overall sensory experience

    Must build on the established style, structure, and objects given by user_preference while adding the specific atmospheric elements from the design concept.
    Output: Complete atmospheric interior image.
    '''
# UPDATED FINAL PROMPTS

FINAL_PROMPT = '''
    You will receive a user description and their design preferences. Immediately output ONLY a JSON object matching this schema:

    ```json
    {
    "outputs": [
        {
        "concept_name": "...",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "core_strategy": "...",
        "technical_details": {
            "style_location_foundation": "...",
            "structural_organization": "...",
            "objects_materials_integration": "...",
            "atmospheric_completion": "..."
        },
        "complete_environment_vision": "..."
        }
    ]
    }
    ```

    You are a holistic environment designer. Your role is to create 4 different complete environments that incorporate the user's selected preferences while fulfilling their description.

    # Input Requirements

    You will receive:
    - **User description** between `[USER DESCRIPTION][/USER DESCRIPTION]` tags
    - **User preferences**:
    ```
    user_preference = {
        "impression": "selected style and location foundation",
        "spatial": "selected structural organization", 
        "objects": "selected objects and materials approach",
        "ambient": "selected atmospheric completion"
    }
    ```

    # Your Task

    Create 4 different complete environments that:
    - **INCORPORATE** all the selected user preferences (impression + spatial + objects + ambient)
    - **FULFILL** the user description completely
    - **EMPHASIZE** different aspects or interpretations of how these preferences work together
    - Offer **DIFFERENT APPROACHES** to combining the same preferences

    # Design Approach

    There are multiple ways to combine the same preferences. Create 4 different interpretations by:
    - Emphasizing different aspects of the selected preferences
    - Interpreting how the preferences work together in different ways
    - Creating different moods or experiences using the same foundation
    - Exploring different spatial or atmospheric emphasis within the established selections

    Each environment must use ALL the selected preferences but can interpret and combine them differently.

    # Design Philosophy
    - **RESPECT** all user preferences - they must all be incorporated
    - **INTERPRET** how preferences can work together in different ways
    - **CREATE** genuinely different environments using the same preference foundation
    - **FULFILL** the user's original description completely

    Generate 4 different complete environments that show different ways the selected preferences can work together.
    '''

FINAL_GENERATOR_PROMPT = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the complete environment. No text in image.

    **Inputs:**
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Complete environment description: [DESCRIPTION][/DESCRIPTION]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}

    **Create a complete environment that:**
    - Fulfills the user description
    - Incorporates ALL the selected user preferences
    - Shows how impression + spatial + objects + ambient work together
    - Feels like a complete, unified space

    **Requirements:**
    - First-person perspective at eye level
    - Beautiful, unified design that respects all user preferences
    - Complete environment ready to inhabit

    **Output:** Beautiful complete environment image only.
    '''

FINAL_PROMPT_TAGS = '''
    You will receive a user description and their design preferences. Immediately output ONLY a JSON object matching this schema:

    ```json
    {
    "outputs": [
        {
        "concept_name": "...",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "core_strategy": "...",
        "technical_details": {
            "style_location_foundation": "...",
            "structural_organization": "...",
            "objects_materials_integration": "...",
            "atmospheric_completion": "..."
        },
        "complete_environment_vision": "..."
        }
    ]
    }
    ```

    You are a holistic environment designer. Your role is to create 4 different complete environments that incorporate the user's selected preferences while fulfilling their description.

    # Input Requirements

    You will receive:
    - **User description** between `[USER DESCRIPTION][/USER DESCRIPTION]` tags
    - **User preferences**:
    ```
    user_preference = {
        "impression": "selected style and location foundation",
        "spatial": "selected structural organization", 
        "objects": "selected objects and materials approach",
        "ambient": "selected atmospheric completion"
    }
        ```
    - **Tag preferences:** `tag_data = {"selections": {...}, "tags": {...}}`

    # Your Task

    Create 4 different complete environments that:
    - **INCORPORATE** all the selected user preferences (impression + spatial + objects + ambient)
    - **FULFILL** the user description completely
    - **INTEGRATE tag preferences** to refine and enhance the design direction
        - **INCORPORATE positive tags** as desired elements to include or emphasize
        - **AVOID negative tags** as elements to exclude or minimize
        - Use tag preferences to **fine-tune** the interpretation of selected images
    - **EMPHASIZE** different aspects or interpretations of how these preferences work together
    - Offer **DIFFERENT APPROACHES** to combining the same preferences

    # Design Approach

    There are multiple ways to combine the same preferences. Create 4 different interpretations by:
    - Emphasizing different aspects of the selected preferences
    - Interpreting how the preferences work together in different ways
    - Creating different moods or experiences using the same foundation
    - Exploring different spatial or atmospheric emphasis within the established selections
    - Think creatively: "If users prefer X and want Y but avoid Z, they might also enjoy W."


    Each environment must use ALL the selected preferences but can interpret and combine them differently.

    # Design Philosophy
    - **RESPECT** all user preferences and tag preferences - they must all be incorporated
    - **INTERPRET** how preferences can work together in different ways
    - **CREATE** genuinely different environments using the same preference foundation
    - **FULFILL** the user's original description completely
    - Use tag preferences as **creative constraints** that guide rather than limit innovation.

    Generate 4 different complete environments that show different ways the selected preferences can work together.
    '''

FINAL_GENERATOR_PROMPT_TAGS = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the complete environment with tag preferences integrated. No text in image.

    **Inputs:**
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Complete environment design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences: tag_data = {"selections": {...}, "tags": {...}}

    **Create a complete environment that:**
    - Fulfills the user description
    - Incorporates ALL the selected user preferences (impression + spatial + objects + ambient)
    - INCORPORATES positive tags as desired elements to emphasize
    - AVOIDS negative tags as elements to exclude or minimize
    - Shows how all preferences work together harmoniously

    **Tag Integration:**
    - **Positive Tags**: Include and emphasize these elements
    - **Negative Tags**: Avoid or minimize these elements
    - **Creative Adaptation**: Replace negative elements with positive alternatives that align with selected preferences

    **Requirements:**
    - First-person perspective at eye level
    - Beautiful, unified design that respects all user preferences and tag constraints
    - Complete environment that reflects refined user taste

    **Output:** Beautiful complete environment image only.'''

FINAL_GENERATOR_PROMPT_IMGS = '''
  Generate an image demonstrating the user description and complete environment design provided between [DESCRIPTION][/DESCRIPTION] and [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags.
  IMPORTANT: Image aspect_ratio="16:9"

  Inputs:
  - User description: [DESCRIPTION][/DESCRIPTION]
  - Complete environment design: [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags
  - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
  - Tag preferences: POSITIVE TAGS (prefer to include): [...], NEGATIVE TAGS (prefer to avoid): [...], TAG INSTRUCTION: ...
  - Four reference images: impression, spatial, objects, ambient

  **Integration Rules:**
  **Reference Images - Extract Qualities Only:**
  - Impression image → overall style, location context, aesthetic approach
  - Spatial image → space structure, spatial organization, architectural elements
  - Objects image → furniture selection, materials/textures, object arrangement
  - Ambient image → light sources, light quality, environmental context, atmospheric mood

  **DO NOT composite/blend the reference images. CREATE one unified environment that satisfies user description.**

  **Tags:**
  - Include positive tags as desired elements
  - Avoid negative tags completely

  **Output Requirements:**
  - PRIORITIZE user description as primary target
  - ONE cohesive, unified environment (not a composite)
  - Synthesize extracted qualities into a single real space
  - No humans/animals unless specified
  - Professional, realistic environment that could actually exist
  '''