# SemanticSlider - Environment Design Generator

An AI-powered environment design system that combines concept refinement with preference-based optimization (PBO) and Stable Diffusion XL for iterative image generation.

## Overview

SemanticSlider enables users to:
1. **Generate Initial Concepts** - Create diverse environment designs from text descriptions
2. **Refine Preferences** - Interact with visual tags and concept clusters through an intuitive bubble chart
3. **Optimize Results** - Use PBO to iteratively refine images based on learned preferences

## Architecture

### Backend (FastAPI + Python)
- **Image Generation**: Stable Diffusion XL (SDXL) integration
- **Tag Extraction**: GPT-4o Vision API for visual element detection
- **Concept Clustering**: Agglomerative clustering for semantic grouping
- **Preference Learning**: Preference-Based Optimization (PBO) with Gaussian Process models
- **Session Management**: Persistent storage of images, tags, and user preferences

### Frontend (React)
- **Interactive UI**: Tag selection and concept visualization
- **Bubble Chart**: D3-style force simulation for concept weight visualization
- **Refinement Controls**: Multi-round PBO iteration interface
- **Session Upload**: Load and continue previous sessions

## Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- CUDA-compatible GPU (recommended for SDXL)
- OpenAI API key (for tag extraction)

### Backend Setup

1. Create and activate virtual environment:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"  # Optional
```

4. Run the server:
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Start development server:
```bash
npm start
```

3. Access the application at `http://localhost:3000`

## Usage

### 1. Generate Initial Impressions

- Enter a text description (e.g., "A cozy room")
- System generates 4 diverse concept images
- Tags are automatically extracted from each image

### 2. Refine Preferences

- Click tags on images to mark as positive (like) or negative (dislike)
- Bubble chart updates to show concept weights
- Larger bubbles = higher weight concepts
- Select your favorite image to continue

### 3. PBO Refinement

- System generates 4 new variations using learned preferences
- Each round explores the concept space based on your selections
- Continue iterating until satisfied with results

### 4. Session Management

**Load Existing Session:**
- Select from dropdown on "Load Existing Session" page
- Automatically loads images, tags, and preferences

**Upload Session Folder:**
- Drag and drop session folder containing:
  - `impression/impression.json`
  - `impression/*.png` (images)
  - `impression/visual_tags.json` (optional, for tags)

## Project Structure

```
SemanticSlider/
├── backend/
│   ├── server.py              # FastAPI main server
│   ├── main.py                # Image generation pipeline
│   ├── sdxl_runner.py         # SDXL integration
│   ├── stage_refiner.py       # PBO refinement logic
│   ├── concept_refinement.py  # Tag clustering and concept management
│   ├── tag_extraction.py      # GPT-4o Vision tag extraction
│   ├── pbo.py                 # Preference-Based Optimization
│   ├── tracking.py            # Session tracking utilities
│   └── sessions/              # Generated session data
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main application
│   │   └── components/
│   │       ├── BubbleChart.jsx              # Concept visualization
│   │       ├── ConceptRefinementPanel.jsx   # Tag interaction UI
│   │       ├── RefinementIterationControls.jsx  # PBO controls
│   │       ├── ProgressBar.jsx              # Stage progress indicator
│   │       └── InlineTagDisplay.jsx         # Tag display on images
│   └── public/
│
└── .gitignore
```

## Key Features

### Concept Refinement System
- **Automatic Clustering**: Groups similar visual tags into semantic concepts
- **Weight Tracking**: Learns concept importance from user interactions
- **Real-time Updates**: Bubble chart reflects preference changes instantly

### Preference-Based Optimization
- **Gaussian Process Learning**: Models user preferences in concept space
- **Exploration vs Exploitation**: Balances diversity and refinement
- **Multi-round Iteration**: Convergence toward optimal preferences

### Session Persistence
- **Complete State Saving**: Images, tags, preferences, and tracking data
- **Resumable Sessions**: Continue from any stage
- **Tracking System**: Detailed logs of all generations and selections

## API Endpoints

### Generation
- `POST /api/generate-fast` - Generate initial impression images
- `POST /api/feedback` - Process selection and generate next stage

### Concept Management
- `POST /api/concepts/init` - Initialize concepts from image tags
- `POST /api/concepts/interact` - Handle tag like/dislike
- `POST /api/tags` - Get tags for specific image

### Session Management
- `GET /api/list-sessions` - List available sessions
- `POST /api/load-stage-data` - Load session data
- `POST /api/upload-session` - Upload session folder

### PBO Refinement
- `POST /api/pbo/refine-next-round` - Generate next refinement round
- `POST /api/pbo/refine-from-weights` - Refine from historical weights
- `GET /api/pbo/debug-state` - Inspect PBO state

## Configuration

### Backend Configuration
- `STAGES`: List of pipeline stages (currently `["impression", "impression_refinement"]`)
- `PROMPTS`: Stage-specific prompts for generation
- SDXL model: `stabilityai/stable-diffusion-xl-base-1.0`

### Frontend Configuration
- API proxy configured in `package.json` (`"proxy": "http://localhost:8000"`)
- Default port: 3000

## Development

### Adding New Stages
1. Add stage name to `STAGES` in `backend/server.py`
2. Add prompts to `PROMPTS` dictionary
3. Update `ProgressBar.jsx` with new stage

### Modifying PBO Parameters
- Edit `backend/pbo.py` for GP kernel parameters
- Edit `backend/stage_refiner.py` for proposal strategies

## Troubleshooting

**Import Errors:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version: 3.8+

**SDXL Memory Issues:**
- Reduce batch size in `sdxl_runner.py`
- Use CPU mode if GPU memory insufficient

**Tag Extraction Fails:**
- Verify OpenAI API key is set
- Check API quota and billing

**Session Not Loading:**
- Ensure `impression/impression.json` exists
- Check file permissions in sessions folder

## License

[Add your license here]

## Acknowledgments

- Stable Diffusion XL by Stability AI
- Preference-Based Optimization research
- OpenAI GPT-4o Vision API
