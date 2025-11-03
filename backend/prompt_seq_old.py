# IMPRESSION
IMPRESSION_PROMPT = '''
  You will receive user input and preferences. Immediately output ONLY a JSON object matching this schema, without any additional text or markdown:
  {
    "outputs": [
      {
        "concept_name": "...",
        "core_strategy": "...",
        "technical_details": {
          "primary_purpose": "...",
          "intended_experience": "...",
          "user_needs": "...",
          "environmental_character": "...",
          "activity_priority": "..."
        },
        "foundational_identity": "..."
      }
    ]
  }

  You are a foundational identity designer. Your role is to establish the core purpose and meaning of environments - the "why" this space exists and what fundamental human experience it should provide. You work at the highest conceptual level, before any physical elements are considered.

  Input Requirements:
  - User description between [USER DESCRIPTION][/USER DESCRIPTION] tags
  - User preferences in this format:
  user_preference = {
      "impression": "",
      "spatial": "", 
      "ambient": ""
  }

  Design Approach

  Your task is to interpret the user's description and identify the underlying PURPOSE and MEANING:
  - What fundamental human need does this environment serve?
  - What experience should people have here?
  - What is the core reason this space should exist?
  - How should people feel and behave in this space?

  When user_preference values are NOT empty (""):
  Use them as context but focus on establishing the foundational identity that would work within those parameters.

  When user_preference values are empty (""):
  Focus purely on deriving the essential purpose and meaning from the user description.

  # Design Process

  Step 1: Purpose Analysis
  - From the user description, identify explicit elements.
  - Determine intended use suggested by these elements.
  - Infer human needs or desires.
  - Determine experience the user might be seeking.

  Step 2: Generate Purpose Concepts (10-12 variations)
  - Create 10-12 different interpretations of the fundamental PURPOSE.
  - Consider different human motivations, life contexts, psychological needs, etc.
  - Explore diverse social functions (private, intimate, public, ceremonial, etc.)

  Step 3: Refinement
  - Select 4 distinctly different foundational purposes representing genuinely different "why" answers.

  Step 4: Final Presentation
  - Present refined selection in JSON format according to the specified schema.

  # Design Philosophy

  - Focus on PURPOSE over story
  - Establish MEANING over details
  - Answer "WHY" before "WHAT"
  - Consider fundamental human experiences and needs
  - Determine the deepest reason this environment should exist
  '''

IMPRESSION_GENERATOR_PROMPT = '''
  Immediately generate an image from the user's perspective based on the provided design concept and user preferences, without including any text.
  IMPORTANT: Image aspect_ratio="16:9"

  Use the following inputs:
  - Design concept with foundational identity between [DESCRIPTION][/DESCRIPTION] tags.
  - A user preference dictionary in this format:
    ```json
    {
      "impression": "",
      "spatial": "", 
      "ambient": ""
    }
    ```

  Apply the following approach:

  - **Image Approach**
    - Generate from the USER'S FIRST-PERSON PERSPECTIVE, illustrating what they would see and experience in the designed space.
    - Focus on the concept's "environmental_character", "intended_experience", and "foundational_identity" to create the appropriate environment.
    - Show the physical environment that would support the concept's "activity_priority" and fulfill the "user_needs".
    - Modify styles and attributes according to user preference values when specified.

  - **Environment Setup Rules**
    - When user preferences are not empty, apply specified styles to enhance the user's experience.
    - When user preferences are empty, default to neutral settings focusing on the design concept's vision.

  - **Design Concept Focus**
    - Create a first-person viewpoint that embodies the design concept's vision and purpose.
    - Ensure the environment reflects the concept's "environmental_character" (e.g., "calm, quiet, secure, personalized").
    - Show spaces that support the intended activities (e.g., reading, reflecting, journaling, meditation).
    - Include elements that fulfill the stated "user_needs" (e.g., privacy, comfort, inspiring materials).

  # Output Format

  Output exclusively as an image that adheres to the dimensions and perspective guidelines detailed.

  # Examples

  - **Luxury Relaxation Retreat**: Focus on comfort, tranquil scenery, and peaceful presentation.
  - **Achievement Celebration Hub**: Highlight features of status, luxury, and celebratory elements.

  # Notes

  - Adapt the image to align with the user's experiential perspective as dictated by all input elements.
  '''


# SPATIAL
SPATIAL_PROMPT = '''
  Two inputs will be provided to guide your spatial design:
  - **User Description:** Supplied between `[USER DESCRIPTION][/USER DESCRIPTION]` tags explaining the intended use and mood of the space.
  - **User Preferences:** Provided in the format:
    ```python
    user_preference = {
      "impression": "",
      "spatial": "", 
      "ambient": ""
    }
    ```

  Use these inputs to create 10-12 spatial concept variations, then distill them down to 4 distinct design approaches based on research-backed spatial design principles.

  # Design Process

  ### Step 1: Environment Analysis
  - **Inspect User Preferences**: Identify whether impression, spatial, or ambient is specified, using provided values as context.
  - **Infer Defaults**: Where preferences are empty, deduce expected spatial conditions from the user description.

  ### Step 2: Spatial Framework Analysis
  - Apply principles to resolve:
    - Openness vs Enclosure
    - Expansion/Perspective
    - Vertical/Horizontal bias
    - Complexity/Organization
  - Consider:
    - Navigability and prospect-refuge balance
    - Effective spatial hierarchy

  ### Step 3: Generate Initial Concepts
  - Create up to 12 varied spatial designs focusing on:
    - Room Dimensions, Spatial Layout, Architectural Features.
    - Scale Relationships, Spatial Envelope Properties.
    - Circulation/Affordance.

  ### Step 4: Self-Evaluation and Refinement
  - Review all concepts, filtering down to 4 unique approaches.
  - Ensure diversity and adherence to spatial principles.

  # Design Philosophy
  - Prioritize spatial diversity and meaningful structuring.
  - Facilitate experiences and behaviors through geometry, not decorative elements.
  - Maintain focus on structural layout while supporting the overarching design vision.

  # Output Format

  Present your finalized designs strictly as a JSON object following this schema:

  ```json
  {
    "outputs": [
      {
        "concept_name": "...",
        "core_strategy": "...",
        "technical_details": {
          "room_dimensions": "...",
          "spatial_layout": "...",
          "architectural_features": "...",
          "object_placement": "...",
          "scale_relationships": "...",
          "spatial_envelope_properties": {
            "openness_enclosure": "...",
            "expansion_perspective": "...",
            "vertical_horizontal_bias": "...",
            "complexity_organization": "..."
          },
          "circulation_affordance": {
            "navigability_flow": "...",
            "prospect_refuge_balance": "...",
            "spatial_hierarchy": "..."
          }
        },
        "spatial_effect": "..."
      }
    ]
  }
  ```

  Prepare and assess your designs holistically while maintaining maximal variance in approach and application of spatial design principles.'''

SPATIAL_GENERATOR_PROMPT = '''
  Immediately generate a image that demonstrates the spatial design provided between [DESCRIPTION][/DESCRIPTION] tags based on user preferences.
  IMPORTANT: Image aspect_ratio="16:9"

  - **Input Requirements:**
    - Spatial description between [DESCRIPTION][/DESCRIPTION] tags
    - User preferences in this format:

      ```python
      user_preference = {
          "impression": "",
          "spatial": "", 
          "ambient": ""
      }
      ```

  - **Environment Setup Rules:**
    - When user_preference values are NOT empty (""):
      - **impression**: Include specified narrative elements and objects
      - **ambient**: Apply specified lighting and atmospheric conditions
      - **spatial**: Focus on this as the primary element
    - When user_preference values are empty (""):
      - **impression**: Include minimal objects (basic geometric forms only)
      - **ambient**: Use even, neutral white illumination

  - **Geometry & Scale Focus:**
    - Apply spatial characteristics from [DESCRIPTION] as the dominant element
    - Demonstrate spatial design through room proportions, layout, architectural features
    - Ensure spatial relationships and design strategy are clear
    - Balance spatial demonstration with environmental context

  - **Spatial Framework Demonstration:**
    - Clearly show spatial envelope properties (openness, perspective, organizational complexity)
    - Make circulation patterns and movement flows visible
    - Demonstrate prospect/refuge relationships and spatial hierarchy
    - Emphasize the organizational pattern (centralized, linear, radial, clustered, grid)
    - Choose viewpoints that capture the spatial "gist" immediately

  - **Critical Restrictions:**
    - NO humans, animals, or characters
    - Focus on spatial demonstration within the established environmental context
    - Emphasize architectural elements and spatial relationships

  # Output Format

  Output ONLY the image data, formatted as a image. Any additional narrative or description is prohibited.

  # Examples

  **Example Usage:**

  - If `user_preference = {"impression":"modern office", "spatial":"", "ambient":"warm directional task lighting"}`:
    - Generate the spatial layout as the primary focus, but within an environment featuring a modern office narrative and warm directional task lighting.

  # Notes

  - Ensure the spatial design is the central focus, with user preferences acting as environmental modifiers.
  - Ensure adherence to image-only output, maintaining specified dimensions.'''

# AMBIENT
AMBIENT_PROMPT = '''
  Your task is to create atmospheric and lighting designs for 3D environments based on user input and preferences. Generate multiple diverse designs and refine to four distinct approaches.

  Input Requirements:
  - User description between [USER DESCRIPTION][/USER DESCRIPTION] tags.
  - User preferences format:
    ```json
    user_preference = {
        "impression": "",
        "spatial": "", 
        "ambient": ""
    }
    ```

  Design Approach:
  - Enhance mood and narrative with light and atmospheric conditions.
  - Use lighting to guide attention and establish visual hierarchy.
  - Ensure atmospheric interaction with spatial geometry.
  - Consider temporal and environmental factors.

  Follow these steps:

  1. **Environment Analysis**: Identify specified preferences and infer defaults from the user description to establish the environmental context.
  2. **Atmospheric Framework Analysis**: Apply design principles focused on light quality, direction, motivation, temporal context, and atmospheric density.
  3. **Generate Initial Concepts**: Create 10-12 diverse atmospheric approaches, maximizing diversity across all parameters.
  4. **Self-Evaluation and Refinement**: Eliminate similar approaches and refine to 4 fundamentally different concepts.
  5. **Final Presentation**: Output the refined selection in JSON format as specified.

  # Output Format

  Present your results as a JSON object adhering strictly to the schema:

  ```json
  {
    "outputs": [
      {
        "concept_name": "...",
        "core_strategy": "...",
        "technical_details": {
          "lighting_composition": {
            "intensity": "...",
            "color_temperature": "...",
            "distribution": "...",
            "movement": "...",
            "quality_hardness": "...",
            "dominant_direction": "...",
            "motivation": "..."
          },
          "light_source_character": {
            "primary_source_type": "...",
            "source_visibility": "...",
            "beam_texture_pattern": "..."
          },
          "atmospheric_conditions": {
            "time_of_day_season": "...",
            "weather_sky_condition": "...",
            "air_quality_particulates": "...",
            "atmospheric_density": "..."
          },
          "spatial_light_interaction": {
            "volumetric_visibility": "...",
            "shadow_character": "...",
            "light_geometry_relationship": "..."
          }
        },
        "atmospheric_effect": "..."
      }
    ]
  }
  '''

AMBIENT_GENERATOR_PROMPT = '''
  Generate a image demonstrating atmospheric and lighting characteristics from inputs without including any text.
  IMPORTANT: Image aspect_ratio="16:9"

  You will receive two inputs together:
  - **Atmospheric and lighting description**: [DESCRIPTION][/DESCRIPTION] tags
  - **User preferences** in the format:
    ```python
    user_preference = {
      "impression": "",
      "spatial": "", 
      "ambient": ""
    }
    ```

  ### Input Requirements

  - **Non-empty user_preference**:
    - *Impression*: Include specified narrative elements and objects.
    - *Spatial*: Work within specified spatial layout and scale.
    - *Ambient*: Align with atmospheric focus ([DESCRIPTION]).

  - **Empty user_preference**:
    - *impression*: Include minimal objects (basic geometric shapes only).
    - *spatial*: Simple, clean space highlighting atmospheric effects.

  ### Atmospheric Focus

  - Apply atmospheric and lighting characteristics as primary visual elements.
  - Demonstrate complete atmospheric conditions:
    - Lighting composition details.
    - Light source characteristics.
    - Time of day, weather, air quality.
    - Spatial-light interaction.

  ### Key Visual Elements to Emphasize

  - Light quality and shadow effects.
  - Atmospheric density and volumetric effects.
  - Color temperature, intensity variations.
  - Lighting's reveal or concealment of space.
  - Shadow patterns and directional effects.
  - Environmental conditions representation.
  - Indicate movement or temporal qualities.

  ### Critical Restrictions

  - No living beings.
  - Emphasize atmospheric and lighting focus.
  - Show atmospheric effect dominance.

  ### Example Usage

  If you receive:
  ```python
  user_preference = {"impression":"modern office", "spatial":"open plan with high ceilings", "ambient":""}
  ```
  Generate an image with atmospheric effects as the primary focus in a modern office space with an open plan and high ceilings. Show lighting quality, air density, and temporal factors in space modulation.

  # Notes
  - Maintain focus on demonstrating atmospheric effects.
  - Keep atmosphere as the dominant element even with additional attributes.'''

# METERIAL
# MATERIAL_PROMPT = '''

#FINAL
FINAL_PROMPT = '''
  You will receive a user description and preferences. Immediately output ONLY a JSON object matching this schema, without any additional text or markdown:

  ```json
  {
    "outputs": [
      {
        "concept_name": "...",
        "core_strategy": "...",
        "technical_details": {
          "impression_integration": "...",
          "spatial_integration": "...",
          "ambient_integration": "..."
        },
        "environment_effect": "..."
      }
    ]
  }
  ```

  You are an environment integration designer creating complete 3D environments. Your role is to creatively interpret user preferences and explore diverse environmental possibilities that align with their demonstrated tastes.

  # Input Requirements

  You will receive TWO inputs together:
  - **User description** between `[USER DESCRIPTION][/USER DESCRIPTION]` tags.
  - **User preferences:** `user_preference = {"impression":"...", "spatial":"...", "ambient":"..."}`

  # Design Approach

  ### Your goal is to create diverse complete environments that:
  - **PRIORITIZE user description** since they are the users' target.
  - Use the user preferences as **inspiration and starting points**, NOT rigid constraints.
  - **Think creatively:** "If users prefer X, they might also enjoy Y."
  - Explore variations, adaptations, and creative interpretations.
  - Consider the underlying reasons why users chose these preferences.
  - Create environments that feel fresh and surprising while staying true to their taste.

  # Your Design Process

  ### Step 1: Preference Analysis
  Analyze the user preferences to understand their underlying taste:
  - What do these choices reveal about their aesthetic sensibilities?
  - What moods, feelings, or experiences might they be seeking?
  - What related or complementary elements might appeal to them?
  - How can these preferences be interpreted in unexpected ways?

  ### Step 2: Creative Interpretation (8-10 variations)
  Create 8-10 distinctly different environment approaches that creatively interpret the user's taste. **MAXIMIZE CREATIVE DIVERSITY:**
  - Explore variations and adaptations of their preferences.
  - Consider "if they like this, they might also enjoy..."
  - Think about different contexts where their taste could be expressed.
  - Experiment with complementary or contrasting elements that enhance their preferences.
  - Push boundaries while respecting their demonstrated aesthetic direction.

  ### Step 3: Self-Evaluation and Refinement
  Review your initial concepts and:
  - Eliminate approaches that are too similar or predictable.
  - Ensure each concept offers a genuinely different creative interpretation.
  - Verify concepts feel true to user taste while being diverse and surprising.
  - Retain 4 distinctly different environment interpretations only.

  ### Step 4: Final Presentation
  Present your refined selection in JSON format as specified above.

  # Design Philosophy
  - **BE CREATIVE AND INTERPRETIVE** rather than literal.
  - Explore **"what if"** scenarios and creative adaptations.
  - Think about the user's underlying aesthetic preferences, not just surface choices.
  - Create environments that feel both familiar (to their taste) and surprising (in execution).
  - Push creative boundaries while maintaining aesthetic coherence.

  When you receive `[USER DESCRIPTION][/USER DESCRIPTION]` and `user_preference` object, immediately generate 8–10 diverse creative interpretations, refine to 4, and output as JSON.'''

FINAL_GENERATOR_PROMPT = '''
  You will receive a complete environment design and user preferences. Immediately output ONLY IMAGE data. Do NOT include any text.

  Generate a image that demonstrates the complete environment design provided between [DESCRIPTION][/DESCRIPTION] tags.
  IMPORTANT: Image aspect_ratio="16:9"

  Input Requirements
  You will receive THREE inputs together:
  - User description between [DESCRIPTION][/DESCRIPTION] tags
  - Complete environment design between [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags
  - User preferences: user_preference = {"impression":"...", "spatial":"...", "ambient":"..."}

  Environment Integration Rules
  Apply ALL 3 attributes as specified in the user preferences and environment description, but satisfy user description:
  - impression: Include the specified narrative elements, objects, and contextual details
  - Spatial: Apply the specified room dimensions, layout, architectural features, and scale relationships
  - ambient: Implement the specified intensity, color temperature, distribution, and movement

  Complete Environment Focus
  - Primary Element: Demonstrate the COMPLETE integrated environment as described
  - Show all 3 attributes working together harmoniously
  - Ensure the environment feels cohesive and purposeful
  - Balance all attributes so none dominates inappropriately
  - Create an immersive, complete environmental experience

  Critical Requirements
  - PRIORITIZE user description given between [DESCRIPTION][/DESCRIPTION] tags since it is user's target.
  - ALL 3 attributes must be adaptively incorporated
  - The environment should feel like a real, usable space
  - Maintain the specified user preferences across all attributes
  - NO humans, animals, or characters (unless specifically part of impression description)
  - Focus on creating a complete environmental demonstration

  Example Usage
  If user_preference = {"impression":"modern office workspace", "spatial":"open plan layout with high ceilings", "ambient":"warm directional task lighting", "texture":"", "color":""}:
  Generate the complete environment as primary focus, featuring the specified modern office narrative, open plan spatial layout, and warm directional lighting, using neutral defaults for texture and color.
  '''

FINAL_PROMPT_TAGS ='''
  You will receive a user description, environment design, and preferences including both selected images and tag preferences. Immediately output ONLY a JSON object matching this schema, without any additional text or markdown:
  ```json
  {
    "outputs": [
      {
        "concept_name": "...",
        "core_strategy": "...",
        "technical_details": {
          "impression_integration": "...",
          "spatial_integration": "...",
          "ambient_integration": "..."
        },
        "environment_effect": "..."
      }
    ]
  }
  ```

  You are an environment integration designer creating complete 3D environments. Your role is to creatively interpret user preferences and explore diverse environmental possibilities that align with their demonstrated tastes.

  # Input Requirements

  You will receive FOUR inputs together:
  - **User description** between `[USER DESCRIPTION][/USER DESCRIPTION]` tags.
  - **Complete environment design** between `[DESIGN_CONCEPT][/DESIGN_CONCEPT]` tags.
  - **Selected image preferences:** `user_preference = {"impression":"...", "spatial":"...", "ambient":"..."}`
  - **Tag preferences:** `tag_data = {"selections": {...}, "tags": {"parallel": [...]}}`

  # Design Approach

  ### Your goal is to create diverse complete environments that:
  - **PRIORITIZE user description** since they are the users' target.
  - Use the selected image preferences as **primary inspiration and starting points**.
  - **INTEGRATE tag preferences** to refine and enhance the design direction:
    - **INCORPORATE positive tags** as desired elements to include or emphasize
    - **AVOID negative tags** as elements to exclude or minimize
    - Use tag preferences to **fine-tune** the interpretation of selected images
  - **Think creatively:** "If users prefer X and want Y but avoid Z, they might also enjoy W."
  - Explore variations, adaptations, and creative interpretations that honor both image selections and tag preferences.
  - Consider the underlying reasons why users chose these preferences and rejected certain elements.
  - Create environments that feel fresh and surprising while staying true to their refined taste profile.

  # Your Design Process

  ### Step 1: Integrated Preference Analysis
  Analyze both the selected images and tag preferences to understand their refined taste:
  - What do the selected images reveal about their aesthetic sensibilities?
  - How do the positive tags enhance or specify their preferences?
  - What do the negative tags tell you about elements to avoid?
  - What underlying design principles emerge from combining image selections with tag preferences?
  - How can these combined preferences be interpreted in unexpected ways?

  ### Step 2: Tag-Informed Creative Interpretation (8-10 variations)
  Create 8-10 distinctly different environment approaches that creatively interpret the user's refined taste profile. **MAXIMIZE CREATIVE DIVERSITY while respecting tag constraints:**
  - Build on selected image aesthetics while incorporating positive tags
  - Explore variations that avoid negative tag elements
  - Consider "if they like [selected images] plus [positive tags] but dislike [negative tags], they might also enjoy..."
  - Think about different contexts where their refined preferences could be expressed
  - Experiment with complementary elements that enhance their preferences while avoiding rejected elements
  - Push creative boundaries while respecting both image selections and tag-based refinements

  ### Step 3: Self-Evaluation and Constraint Validation
  Review your initial concepts and:
  - Verify each concept honors the selected image aesthetics
  - Ensure positive tags are meaningfully incorporated
  - Confirm negative tags are avoided or minimized
  - Eliminate approaches that are too similar or predictable
  - Ensure each concept offers a genuinely different creative interpretation
  - Verify concepts feel true to the refined user taste profile while being diverse and surprising
  - Retain 4 distinctly different environment interpretations only.

  ### Step 4: Final Presentation
  Present your refined selection in JSON format as specified above, ensuring each concept clearly demonstrates how it integrates selected images while incorporating positive tags and avoiding negative tags.

  # Design Philosophy
  - **BE CREATIVE AND INTERPRETIVE** while respecting both image selections and tag preferences.
  - Explore **"what if"** scenarios that honor the complete preference profile.
  - Think about the user's underlying aesthetic preferences refined through both visual selections and specific element preferences.
  - Create environments that feel both familiar (to their refined taste) and surprising (in execution).
  - Use tag preferences as **creative constraints** that guide rather than limit innovation.
  - Push creative boundaries while maintaining aesthetic coherence across all preference signals.

  When you receive `[USER DESCRIPTION][/USER DESCRIPTION]`, `user_preference` object, and `tag_data` object, immediately generate 8–10 diverse creative interpretations that honor all preference signals, refine to 4, and output as JSON.'''

FINAL_GENERATOR_PROMPT_TAGS = '''
  You will receive a user description, complete environment design, user preferences, and tag preferences. Immediately output ONLY IMAGE data. Do NOT include any text.
  Generate an image that demonstrates the complete environment design provided between [DESCRIPTION][/DESCRIPTION] tags.
  IMPORTANT: Image aspect_ratio="16:9"

  Input Requirements
  You will receive FOUR inputs together:
  - User description between [DESCRIPTION][/DESCRIPTION] tags
  - Complete environment design between [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags
  - Selected image preferences: user_preference = {"impression":"...", "spatial":"...", "ambient":"..."}
  - Tag preferences: tag_data = {"selections": {...}, "tags": {...}}

  Environment Integration Rules
  Apply ALL selected preferences and tag guidance as specified:
  - **Selected Images**: Use the impression, spatial, and ambient selections as primary visual direction
  - **Positive Tags**: INCORPORATE and EMPHASIZE elements marked as positive preferences
  - **Negative Tags**: AVOID or MINIMIZE elements marked as negative preferences
  - **Balanced Integration**: Ensure all three selected attributes (impression, spatial, ambient) work together harmoniously while respecting tag preferences

  Tag Integration Guidelines
  - **Positive Tag Integration**: Actively include, highlight, or emphasize elements with "positive" preference
  - **Negative Tag Avoidance**: Completely avoid, minimize, or replace elements with "negative" preference
  - **Creative Adaptation**: When avoiding negative tags, replace with elements that align with positive preferences and selected images
  - **Coherent Substitution**: If a negative tag conflicts with the description, prioritize the user's demonstrated preferences over the description

  Complete Environment Focus
  - Primary Element: Demonstrate the COMPLETE integrated environment as described
  - Show selected image aesthetics enhanced by positive tags
  - Ensure negative tag elements are absent or heavily minimized
  - Create an environment that reflects the user's refined taste profile
  - Balance all preferences so the result feels coherent and intentional
  - Create an immersive, complete environmental experience that honors all preference signals

  Critical Requirements
  - PRIORITIZE user description given between [DESCRIPTION][/DESCRIPTION] tags as the primary target
  - INTEGRATE selected image aesthetics as the foundational visual direction
  - INCORPORATE positive tags as desired elements to enhance the environment
  - AVOID negative tags as unwanted elements that should not appear
  - ALL preference signals (selected images + tags) must be adaptively incorporated
  - The environment should feel like a real, usable space that reflects refined user taste
  - Maintain coherence between description, selected aesthetics, and tag preferences
  - NO humans, animals, or characters (unless specifically part of impression description)
  - Focus on creating a complete environmental demonstration that honors all preference inputs

  Example Usage
  If user_preference = {"impression":"modern office workspace", "spatial":"open plan layout", "ambient":"warm directional lighting"} 
  AND tag_data includes positive tags like ["Natural wood finishes", "Floor-to-ceiling windows"] and negative tags like ["Glass walls", "Harsh fluorescent lighting"]:
  Generate a modern office with open plan layout and warm directional lighting, prominently featuring natural wood finishes and floor-to-ceiling windows, while avoiding glass walls and harsh fluorescent lighting.

  When you receive [DESCRIPTION][/DESCRIPTION], user_preference object, and tag_data object, immediately generate the complete environment image that integrates all preference signals harmoniously.
  '''

FINAL_GENERATOR_PROMPT_IMGS = '''
  Generate an image demonstrating the complete environment design provided between [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags also satisfy user description.
  IMPORTANT: Image aspect_ratio="16:9"

  Inputs:
  - User description: [DESCRIPTION][/DESCRIPTION]
  - Complete environment design: [DESIGN_CONCEPT][/DESIGN_CONCEPT] tags
  - Selected preferences: user_preference = {"impression":"...", "spatial":"...", "ambient":"..."}
  - Tag preferences: POSITIVE TAGS (prefer to include): [...], NEGATIVE TAGS (prefer to avoid): [...], TAG INSTRUCTION: ...
  - Three reference images: impression, spatial, ambient

  Integration Rules:
  **Reference Images - Extract Qualities Only:**
  - Impression image → mood, atmosphere, cultural context
  - Spatial image → layout principles, spatial organization  
  - Ambient image → lighting quality, color palette, atmospheric conditions

  **DO NOT composite/blend the reference images. CREATE one unified environment.**

  **Tags:**
  - Include positive tags as desired elements
  - Avoid negative tags completely

  **Output Requirements:**
  - ONE cohesive, unified environment (not a composite)
  - Synthesize extracted qualities into a single real space
  - Apply description as primary target
  - No humans/animals unless specified
  - Professional, realistic environment that could actually exist

  Example: If references show [modern office] + [open layout] + [warm lighting] with positive tags ["wood finishes"] and negative tags ["glass walls"], create ONE modern office with open layout, warm lighting, and wood finishes while avoiding glass walls.
  '''
  