# Evaluation Prototype

This evaluation prototype allows users to explore predefined sessions and generate semantic sliders without going through the refinement stage.

## Overview

The prototype uses **exactly the same interface design and layout** as the FULL implementation, including:
- 2x2 image grid with clickable images for selection
- Bubble chart for concept visualization and interaction
- Tag display with positive/negative preference indicators
- Tag sidebar and JSON panel

### Simplified Flow:
1. **Landing** - Select a predefined session
2. **Exploration** - Same as FULL implementation with bubble chart
3. **Slider Generation** - Uses exploration weights directly (skips refinement)

## Directory Structure

```
eval/
├── backend/
│   ├── eval_server.py      # Main eval server (port 8001)
│   ├── eval_utils.py       # Utility functions
│   └── __init__.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Simplified eval app
│   │   ├── components/     # Copied from main frontend
│   │   └── index.js
│   ├── public/
│   ├── package.json        # Runs on port 3001
│   └── craco.config.js
├── predefined_input/       # Predefined sessions to load
│   └── cozy_bedroom_sample/
├── session_logs/           # Eval session logs (created per run)
└── README.md
```

## Quick Start

### 1. Start the Eval Backend

```bash
cd eval/backend
conda activate apl
python eval_server.py
```

The server runs on port 8001.

### 2. Start the Eval Frontend

```bash
cd eval/frontend
npm install
npm start
```

The frontend runs on port 3001. Open http://localhost:3001 in your browser.

### 3. Add Predefined Sessions

Add session folders to `eval/predefined_input/`. Each session needs:

```
session_name/
├── final_selection.json    # Contains adjective, location, descriptor
├── preferences.json        # Can be empty: {}
└── impression/
    ├── impression.json     # Concept definitions
    ├── visual_tags.json    # Tags for each image
    ├── concept_weights.json # Initial concept weights
    └── impression_*.png    # Generated images (4 images)
```

## Flow Comparison

| Stage | FULL Implementation | Eval Prototype |
|-------|---------------------|----------------|
| Load Session | Mode 2 - Upload/Select | Load from predefined_input |
| Exploration | Full interaction | Same |
| After Exploration | Save weights, go to refinement | Save weights, auto-generate final_selection |
| Refinement | Multiple PBO rounds | **SKIPPED** |
| Slider Generation | Uses refinement weights | Uses exploration weights directly |

## API Endpoints (Eval-specific)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/eval/predefined-sessions` | GET | List available predefined sessions |
| `/api/eval/load-session` | POST | Load a predefined session |
| `/api/eval/skip-to-slider` | POST | Skip refinement, generate final_selection from exploration weights |
| `/api/eval/status/{session_id}` | GET | Get session status |

## Session Logging

Each evaluation run creates a log folder in `session_logs/` with:
- User ID and timestamp in folder name
- Copied session data
- `eval_log.json` tracking events and timing
- Updated `concept_weights.json` after exploration
- Generated `final_selection.json` from exploration weights

