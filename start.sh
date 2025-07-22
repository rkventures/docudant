#!/bin/bash
# Start FastAPI backend (e.g., for /upload) on port 8000
uvicorn app:api --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on port 10000 (Render default)
streamlit run app.py --server.port 10000 --server.enableCORS false
