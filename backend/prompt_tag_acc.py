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
# v3: added a mini “adaptation reasoning step” before each generation
IMPRESSION_GENERATOR_PROMPT_v3 = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") based on the provided style and location foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Foundation design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    ADAPTATION REASONING (DO THIS FIRST, NO IMAGE YET)
    1. Identify all visual traits in the design concept that match or resemble any negative tags.
    2. For each conflicting trait, choose a visually distinct alternative that still serves the concept’s purpose, style intent, and location context.
    3. Identify any positive tags that fit naturally without breaking the concept’s coherence — note where they could be integrated.
    4. Keep the concept’s core purpose, function, and experience untouched.

    GENERATION RULES
    1. Preserve the concept’s style direction, location context, and design intent exactly as described.
    2. Apply replacements from the adaptation reasoning step — negative tag traits must be absent.
    3. Include positive tags naturally where possible.
    4. Ensure the final style foundation is clear and fully coherent.

    OUTPUT
    A photorealistic, first-person interior image that preserves the style/location foundation while visually adapting it according to tag preferences.
    '''
    
SPATIAL_GENERATOR_PROMPT_v3 = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the space structure within the established style foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Spatial design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    ADAPTATION REASONING (DO THIS FIRST, NO IMAGE YET)
    1. Identify spatial/structural traits in the concept that match or resemble negative tags.
    2. For each, choose a functional alternative that maintains the concept’s spatial flow, zone division, and architectural intent but avoids the negative trait.
    3. Note any positive spatial tags that can be included naturally without disrupting function.

    GENERATION RULES
    1. Keep the structure, zone division, architectural elements, and spatial flow exactly as in the concept.
    2. Remove/replace negative tag traits with alternatives from the reasoning step.
    3. Include positive tags only where they fit naturally.
    4. Maintain style/location alignment from the Impression phase.

    OUTPUT
    A photorealistic, first-person interior image that preserves the spatial concept while visually adapting to tag preferences.
    '''
    
OBJECTS_GENERATOR_PROMPT_v3 = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the objects and materials within the established design. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Objects/materials design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    ADAPTATION REASONING (DO THIS FIRST, NO IMAGE YET)
    1. Identify any furniture, decor, materials, or textures in the concept that match or resemble negative tags.
    2. Replace each with a visually distinct alternative that retains the same functional purpose and fits the style.
    3. Identify positive tags that can be naturally integrated.

    GENERATION RULES
    1. Preserve object arrangement, functional organization, and overall material strategy from the concept.
    2. Apply replacements from the reasoning step — no negative tag traits remain.
    3. Include positive tags only if they fit naturally.
    4. Ensure visual coherence with the Impression and Spatial phases.

    OUTPUT
    A photorealistic, first-person interior image showing objects and materials that preserve the concept while adapting to tag preferences.
    '''
    
AMBIENT_GENERATOR_PROMPT_v3 = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the complete atmospheric experience. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Atmospheric design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    ADAPTATION REASONING (DO THIS FIRST, NO IMAGE YET)
    1. Identify lighting, weather, and mood elements in the concept that match or resemble negative tags.
    2. Replace each with a visually distinct alternative that still achieves the intended atmosphere.
    3. Identify positive tags that can be naturally included.

    GENERATION RULES
    1. Preserve the concept’s intended mood, sensory feel, and environmental context.
    2. Apply replacements from the reasoning step — no negative tag traits remain.
    3. Include positive tags where natural.
    4. Maintain coherence with all previous phases.

    OUTPUT
    A photorealistic, first-person interior image that preserves the atmospheric concept while adapting it to tag preferences.
    '''


# v2: Design concept = anchor, Negative tags = strict visual adaptation rules, Positive tags = soft influence
IMPRESSION_GENERATOR_PROMPT = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") based on the provided style and location foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Foundation design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    PRIORITY RULES
    1. Preserve the design concept’s core intent, style direction, location context, and design intent exactly as described. 
    - Do not alter the purpose, core function, or experiential goals.
    - Maintain the concept’s defining characteristics unless a visual trait conflicts with a negative tag.

    2. Adapt the concept visually according to tag preferences:
    - **Negative tags**: Completely remove visual traits linked to them (objects, motifs, colors, textures, lighting cues, atmospheres).
        * Replace with visually distinct alternatives that still serve the concept’s purpose and style intent.
        * Avoid close visual relatives of the negative tag’s style family.
    - **Positive tags**: Include and subtly emphasize if they fit naturally.

    3. Ensure the adapted style foundation still clearly communicates:
    - The overall style direction.
    - The location context.
    - The style characteristics and design intent — expressed through tag-compliant aesthetics.

    CHECKLIST BEFORE FINALIZING
    - Does the image clearly show the concept’s style and location? → Must be YES.
    - Does it completely avoid all negative tag traits? → Must be YES.
    - Are alternatives visually distinct from negative tags? → Must be YES.
    - Do positives appear naturally where possible? → Prefer YES.

    OUTPUT
    A photorealistic, first-person interior image that preserves the style/location foundation while visually adapting it to tag preferences.
    '''

SPATIAL_GENERATOR_PROMPT = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the space structure within the established style foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Spatial design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    PRIORITY RULES
    1. Preserve the spatial concept’s structure, division of zones, architectural elements, and spatial flow exactly as described.
    - Do not change layout logic or the way space supports activities.
    - Keep alignment with the Impression phase style and location.

    2. Adapt structural expression visually according to tag preferences:
    - **Negative tags**: Remove or alter visual/structural traits tied to them (e.g., cramped, segmented, obstructed).
        * Replace with contrasting spatial solutions that retain the same functional layout intent.
        * Avoid spatial characteristics too similar to the negative tags.
    - **Positive tags**: Integrate and highlight where they fit naturally.

    3. Ensure the image communicates:
    - The intended space organization.
    - Division of zones and architectural elements.
    - Spatial flow — all consistent with the concept but filtered through tag compliance.

    CHECKLIST BEFORE FINALIZING
    - Is the structure exactly as per concept intent? → Must be YES.
    - Does it avoid all negative tag-related spatial traits? → Must be YES.
    - Are replacements distinct but functional? → Must be YES.
    - Do positives appear naturally where possible? → Prefer YES.

    OUTPUT
    A photorealistic, first-person interior image that maintains the structural concept while visually adapting it according to tag preferences.
    '''

OBJECTS_GENERATOR_PROMPT = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the objects and materials within the established design. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Objects/materials design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    PRIORITY RULES
    1. Preserve the object selection, arrangement logic, and material strategy as described in the concept.
    - Keep objects that fulfill the intended activities and stylistic role.
    - Maintain functional organization and the concept’s intended material character unless it directly conflicts with a negative tag.

    2. Adapt visual/material expression according to tag preferences:
    - **Negative tags**: Remove objects/materials/textures linked to them.
        * Replace with visually distinct alternatives that retain the same functional and stylistic role.
        * Avoid material/object types closely related to the negative tag’s aesthetic.
    - **Positive tags**: Include when they fit naturally.

    3. Ensure the adapted object/material choices still convey:
    - The intended function.
    - The established style identity.
    - Coherence with previous phases.

    CHECKLIST BEFORE FINALIZING
    - Do objects and materials match concept intent? → Must be YES.
    - Are all negative tag traits absent? → Must be YES.
    - Are alternatives distinct and coherent? → Must be YES.
    - Do positives appear naturally where possible? → Prefer YES.

    OUTPUT
    A photorealistic, first-person interior image showing objects and materials that preserve the concept while adapting to tag preferences.
    '''

AMBIENT_GENERATOR_PROMPT = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the complete atmospheric experience. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Atmospheric design concept: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        + Positive tags (OPTIONAL): [POSITIVE_TAGS][/POSITIVE_TAGS]
        + Negative tags (HARD AVOID): [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    PRIORITY RULES
    1. Preserve the atmospheric concept’s intent for mood, sensory feel, and environmental context.
    - Keep time of day, weather, and light source logic unless they directly conflict with a negative tag.

    2. Adapt lighting/atmospheric visuals according to tag preferences:
    - **Negative tags**: Remove all lighting/atmosphere traits tied to them.
        * Replace with visually distinct mood/lighting solutions that still express the concept’s intended feel.
        * Avoid closely related lighting moods or atmospheric conditions.
    - **Positive tags**: Include when natural.

    3. Ensure the adapted atmosphere still communicates:
    - The intended mood and sensory experience.
    - The environmental context.
    - Coherence with all prior phases.

    CHECKLIST BEFORE FINALIZING
    - Is the mood/environment as per concept intent? → Must be YES.
    - Are all negative tag traits absent? → Must be YES.
    - Are alternatives distinct but concept-aligned? → Must be YES.
    - Do positives appear naturally where possible? → Prefer YES.

    OUTPUT
    A photorealistic, first-person interior image that preserves the atmospheric concept while visually adapting to tag preferences.
    '''


# v1:naive emphasis on tag preferences
IMPRESSION_GENERATOR_PROMPT_v1 = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the style and location foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Foundation design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - Tag preferences:
        * Positive tags (PREFER to include): [POSITIVE_TAGS][/POSITIVE_TAGS]
        * Negative tags (MUST AVOID + EXPLORE ALTERNATIVES): [NEGATIVE_TAGS][/NEGATIVE_TAGS]


    Create an image that establishes the style foundation:
    - FIRST PRIORITY: Completely avoid all negative tags and replace with alternative approaches
    - SECOND PRIORITY: Shows the design concept clearly 
    - THIRD PRIORITY: Prefer to include and emphasize all positive tags
    - Shows the OVERALL STYLE clearly (filtered through tag preferences)
    - Demonstrates the LOCATION CONTEXT (adapted to tag preferences)
    - Expresses the STYLE CHARACTERISTICS (modified by tag preferences)
    - Reflects the DESIGN INTENT (constrained by tag requirements)

    Preference Integration Guidelines (MANDATORY):
    - If positive tags exist: These elements should be emphasized throughout the design
    - If negative tags exist: These elements MUST BE COMPLETELY AVOIDED. Replace with opposite alternatives (e.g., "cluttered aesthetic" → AVOID clutter, USE "clean minimalism"; "cold atmosphere" → AVOID coldness, USE "warm inviting ambiance"; "harsh lighting" → AVOID harsh lights, USE "soft gentle lighting")
    - Tag preferences override design concept if there's conflict
    - Every design decision must pass the tag filter: "Does this respect positive tags? Does this avoid negative tags?"

    Focus on beautiful representation that STRICTLY adheres to tag preferences.
    Output: Beautiful interior image that demonstrates clear tag compliance.
    '''

SPATIAL_GENERATOR_PROMPT_v1 = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the spatial structure within the established style foundation. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Structural design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        * MUST INCLUDE these preferred elements: [POSITIVE_TAGS][/POSITIVE_TAGS]
        * MUST AVOID and replace with alternatives: [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    CRITICAL: Tag preferences are the PRIMARY DIRECTIVE. The spatial design MUST respect these preferences above all else.

    Create an image showing:
    - FIRST PRIORITY: Completely avoid all negative spatial tags and implement opposite spatial solutions
    - SECOND PRIORITY: Demonstrate the design concept in the image
    - THIRD PRIORITY: Prefer to include and emphasize all positive spatial tags
    - The SPACE STRUCTURE organized according to tag preferences
    - The SPACE DIVISION that respects tag requirements
    - The ARCHITECTURAL ELEMENTS filtered through tag preferences
    - The SPATIAL FLOW constrained by tag compliance

    Preference Integration Guidelines (MANDATORY):
    - If positive tags exist: These spatial arrangements Should be featured and emphasized
    - If negative tags exist: These spatial elements MUST BE COMPLETELY AVOIDED. Use opposite approaches (e.g., "cramped layout" → AVOID cramped spaces, USE "spacious open layout"; "disconnected spaces" → AVOID separation, USE "seamless flow"; "blocked circulation" → AVOID obstacles, USE "clear open pathways")
    - Tag preferences override structural design if there's conflict
    - Every spatial decision must pass: "Does this include positive spatial tags? Does this avoid negative spatial tags?"

    Must work within user_preference foundation while STRICTLY adhering to spatial tag requirements.
    Output: Interior showing spatial structure with clear tag compliance.
    '''

OBJECTS_GENERATOR_PROMPT_v1 = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the objects and materials within the established design. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Objects design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        * MUST INCLUDE these preferred elements: [POSITIVE_TAGS][/POSITIVE_TAGS]
        * MUST AVOID and replace with alternatives: [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    CRITICAL: Tag preferences are the PRIMARY DIRECTIVE. Object and material choices MUST respect these preferences above all else.

    Create an image showing:
    - FIRST PRIORITY: Completely exclude all negative object/material tags and use opposite alternatives
    - SECOND PRIORITY: Demonstrate the design concept in the image
    - THIRD PRIORITY: Prefer to include and emphasize all positive object/material tags
    - APPROPRIATE OBJECTS selected based on tag requirements
    - OBJECT ARRANGEMENT that demonstrates tag compliance
    - MATERIALS & TEXTURES filtered through tag preferences
    - FUNCTIONAL ORGANIZATION constrained by tag adherence

    Preference Integration Guidelines (MANDATORY):
    - If positive tags exist: These objects/materials Should be emphasized
    - If negative tags exist: These objects/materials MUST BE COMPLETELY EXCLUDED. Use opposite alternatives (e.g., "modern appliances" → AVOID modern style, USE "vintage/traditional appliances"; "plastic materials" → AVOID plastic, USE "natural wood/stone/metal"; "busy patterns" → AVOID busy designs, USE "clean solid colors or minimal patterns")
    - Tag preferences override object design if there's conflict
    - Every object/material choice must pass: "Does this include positive tags? Does this completely avoid negative tags?"

    Must work within established foundation while STRICTLY adhering to object/material tag requirements.
    Output: Interior showing objects and materials with clear tag compliance.
    '''

AMBIENT_GENERATOR_PROMPT_v1 = '''
    Generate a first-person interior view (aspect_ratio="16:9") showing the complete atmospheric experience. No text.

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Atmospheric design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Tag preferences:
        * MUST INCLUDE these preferred elements: [POSITIVE_TAGS][/POSITIVE_TAGS]
        * MUST AVOID and replace with alternatives: [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    CRITICAL: Tag preferences are the PRIMARY DIRECTIVE. Atmospheric elements MUST respect these preferences above all else.

    Create an image showing the complete design with atmospheric completion:
    - FIRST PRIORITY: Completely avoid all negative atmospheric tags and implement opposite lighting/mood solutions
    - SECOND PRIORITY: Demonstrate the design concept in the image
    - THIRD PRIORITY: Prefer to include and emphasize all positive atmospheric tags
    - LIGHT SOURCES chosen based on tag requirements
    - LIGHT QUALITY filtered through tag preferences
    - ENVIRONMENTAL CONTEXT constrained by tag compliance
    - ATMOSPHERIC MOOD that demonstrates tag adherence

    Preference Integration Guidelines (MANDATORY):
    - If positive tags exist: These atmospheric elements Should be emphasized
    - If negative tags exist: These atmospheric elements MUST BE COMPLETELY AVOIDED. Use opposite approaches (e.g., "harsh overhead lighting" → AVOID harsh/overhead, USE "soft accent/ambient lighting"; "artificial illumination" → AVOID artificial, USE "natural window light"; "sterile atmosphere" → AVOID sterile feeling, USE "warm cozy ambient glow")
    - Tag preferences override atmospheric design if there's conflict
    - Every lighting/atmospheric choice must pass: "Does this include positive atmospheric tags? Does this completely avoid negative atmospheric tags?"

    Must build on established foundation while STRICTLY adhering to atmospheric tag requirements.
    Output: Complete atmospheric interior with clear tag compliance.
    '''

# UPDATED FINAL PROMPTS (generate one concept a time)

FINAL_PROMPT_one_concept = '''
    You are a holistic environment designer. Your role is to create ONE complete environment that incorporates the user's selected preferences while fulfilling their description.

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
    - **Tag preferences**:
        * INCORPORATE these preferred elements: [POSITIVE_TAGS][/POSITIVE_TAGS]
        * AVOID and replace with alternatives: [NEGATIVE_TAGS][/NEGATIVE_TAGS]

    # Your Task

    Create ONE complete environment that:
    - **INCORPORATES** all the selected user preferences (impression + spatial + objects + ambient)
    - **FULFILLS** the user description completely
    - **RESPECTS** tag preferences from previous selections if not empty
    - **INTEGRATES** all elements into one cohesive design vision

    # Tag Integration Instructions

    - If positive tags exist: These elements MUST be prominently featured and emphasized throughout the complete design
    - If negative tags exist: These elements MUST BE COMPLETELY AVOIDED. Replace with alternative approaches that serve the same purpose
    - All tag preferences must be respected while maintaining the user_preference foundation
    - Create one unified environment that demonstrates successful tag compliance

    # Design Philosophy

    - **RESPECT** all user preferences - they must all be incorporated
    - **INTEGRATE** preferences into one seamless, unified vision
    - **AVOID** negative tags completely while emphasizing positive tags
    - **CREATE** a buildable, realistic complete environment
    - **FULFILL** the user's original description completely

    Generate ONE complete environment concept that shows how all selected preferences work together while respecting learned tag preferences.

    Output format (JSON only):
    {
    "output": {
        "concept_name": "Brief descriptive name for the complete design",
        "user_description": "<copy text from [USER DESCRIPTION] verbatim>",
        "core_strategy": "Overall approach to integrating all preferences and tag feedback",
        "technical_details": {
            "style_location_foundation": "How the selected impression manifests with tag preferences",
            "structural_organization": "How the selected spatial approach works with tag preferences",
            "objects_materials_integration": "How selected objects/materials incorporate tag preferences",
            "atmospheric_completion": "How selected ambient approach respects tag preferences"
        },
        "complete_environment_vision": "How all elements work together as one unified, tag-compliant design"
    }
    }
    '''

FINAL_GENERATOR_PROMPT_one_concept = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the complete environment. No text in image.

    **Inputs:**
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Complete environment description: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - **Tag preferences**:
        * INCORPORATE these preferred elements: [POSITIVE_TAGS][/POSITIVE_TAGS]
        * AVOID and replace with alternatives: [NEGATIVE_TAGS][/NEGATIVE_TAGS]
 
    Create an image showing the complete unified design:
    - FIRST PRIORITY: Demonstrate the design concept in the image
    - SECOND PRIORITY: Completely avoid all negative tags accumulated across ALL levels
    - THIRD PRIORITY: Prominently feature all positive tags from ALL levels
    - Shows the COMPLETE STYLE fully realized and integrated
    - Demonstrates the SPATIAL STRUCTURE as built and furnished
    - Features the SELECTED OBJECTS AND MATERIALS in their final arrangement
    - Displays the CHOSEN LIGHTING AND ATMOSPHERE completing the experience
    - Presents a COHESIVE, UNIFIED design where all elements work together seamlessly

    Tag Integration Instructions (MANDATORY):
    - If negative tags exist: These elements MUST BE COMPLETELY AVOIDED across all aspects of the design
    - If positive tags exist: These elements should be included and emphasized throughout the complete design

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

# FINAL STAGE PROMPTS FOR CUMULATIVE TAGS MODE
FINAL_PROMPT_CUMULATIVE = '''
    You will receive a user description and their design preferences from cumulative tags mode. Immediately output ONLY a JSON object matching this schema:

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

FINAL_GENERATOR_PROMPT_CUMULATIVE = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the complete environment. No text in image.

    **Inputs:**
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Complete environment description: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
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

FINAL_PROMPT_CUMULATIVE_TAGS = '''
    You will receive a user description and their design preferences from cumulative tags mode. Immediately output ONLY a JSON object matching this schema:

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
    - **Cumulative tag preferences:** `tag_data = {"impression": {...}, "spatial": {...}, "objects": {...}, "ambient": {...}}`

    # Your Task

    Create 4 different complete environments that:
    - **INCORPORATE** all the selected user preferences (impression + spatial + objects + ambient)
    - **FULFILL** the user description completely
    - **INTEGRATE cumulative tag preferences** to refine and enhance the design direction
        - **INCORPORATE positive tags** from each stage as desired elements to include or emphasize
        - **AVOID negative tags** from each stage as elements to exclude or minimize
        - Use cumulative tag preferences to **fine-tune** the interpretation of selected images
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
    - **RESPECT** all user preferences and cumulative tag preferences - they must all be incorporated
    - **INTERPRET** how preferences can work together in different ways
    - **CREATE** genuinely different environments using the same preference foundation
    - **FULFILL** the user's original description completely
    - Use cumulative tag preferences as **creative constraints** that guide rather than limit innovation.

    Generate 4 different complete environments that show different ways the selected preferences can work together.
    '''

FINAL_GENERATOR_PROMPT_CUMULATIVE_TAGS = '''
    Generate a beautiful first-person interior view (aspect_ratio="16:9") showing the complete environment with cumulative tag preferences integrated. No text in image.

    **Inputs:**
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Complete environment design: [DESIGN_CONCEPT][/DESIGN_CONCEPT]
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Cumulative tag preferences: tag_data = {"impression": {...}, "spatial": {...}, "objects": {...}, "ambient": {...}}

    **Create a complete environment that:**
    - Fulfills the user description
    - Incorporates ALL the selected user preferences (impression + spatial + objects + ambient)
    - INCORPORATES positive tags from each stage as desired elements to emphasize
    - AVOIDS negative tags from each stage as elements to exclude or minimize
    - Shows how all preferences work together harmoniously

    **Cumulative Tag Integration:**
    - **Positive Tags from Impression**: Include and emphasize these style/location elements
    - **Positive Tags from Spatial**: Include and emphasize these structural/organizational elements
    - **Positive Tags from Objects**: Include and emphasize these furnishing/material elements
    - **Positive Tags from Ambient**: Include and emphasize these atmospheric/lighting elements
    - **Negative Tags from All Stages**: Avoid or minimize these elements completely
    - **Creative Adaptation**: Replace negative elements with positive alternatives that align with selected preferences

    **Requirements:**
    - First-person perspective at eye level
    - Beautiful, unified design that respects all user preferences and cumulative tag constraints
    - Complete environment that reflects refined user taste from all stages

    **Output:** Beautiful complete environment image only.
    '''

FINAL_GENERATOR_PROMPT_CUMULATIVE_IMGS = '''
    Generate an image demonstrating the user description and complete environment design provided between [DESCRIPTION][/DESCRIPTION] and [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags.
    IMPORTANT: Image aspect_ratio="16:9"

    Inputs:
    - User description: [DESCRIPTION][/DESCRIPTION]
    - Complete environment design: [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags
    - User preferences: user_preference = {"impression":"...", "spatial":"...", "objects":"...", "ambient":"..."}
    - Cumulative tag preferences: tag_data = {"impression": {...}, "spatial": {...}, "objects": {...}, "ambient": {...}}
    - Four reference images: impression, spatial, objects, ambient

    **Integration Rules:**
    **Reference Images - Extract Qualities Only:**
    - Impression image → overall style, location context, aesthetic approach
    - Spatial image → space structure, spatial organization, architectural elements
    - Objects image → furniture selection, materials/textures, object arrangement
    - Ambient image → light sources, light quality, environmental context, atmospheric mood

    **DO NOT composite/blend the reference images. CREATE one unified environment that satisfies user description.**

    **Cumulative Tags:**
    - Include positive tags from each stage as desired elements
    - Avoid negative tags from each stage completely

    **Output Requirements:**
    - PRIORITIZE user description as primary target
    - ONE cohesive, unified environment (not a composite)
    - Synthesize extracted qualities into a single real space
    - No humans/animals unless specified
    - Professional, realistic environment that could actually exist
    '''