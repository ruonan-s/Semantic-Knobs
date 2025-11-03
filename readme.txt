# Terminal1: run backend
cd ui-without-pbo/backend
uvicorn server:app --reload --port 8000    

# Terminal2: run frontend
cd ui-without-pbo/frontend
npm start    