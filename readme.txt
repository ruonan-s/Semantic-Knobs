# Terminal1: run backend
cd backend
uvicorn server:app --reload --port 8000

# Terminal2: run frontend
cd frontend
npm start

# PBO Integration Status
Stage 1: ✅ Core PBO class (backend/pbo.py)
Stage 2: ✅ StageRefiner (backend/stage_refiner.py)
Stage 3: ✅ SDXL Integration (backend/sdxl_integration.py, sdxl_runner.py, sdxl_embed_fuser.py)
Stage 4: ✅ Server API Endpoints (4 new endpoints in backend/server.py)

# Testing
# Run PBO tests
conda activate apl
python backend/test_pbo.py
python backend/test_stage_refiner.py
python backend/test_sdxl_integration.py
python backend/test_pbo_endpoints.py

# See STAGE3_HANDOFF.md for Stage 3 details
# See STAGE4_QUICK_START.md for Stage 4 details and API reference    