#!/bin/bash

# Install system and Python dependencies (only runs at container startup)
apt-get update && \
apt-get install -y tesseract-ocr && \
pip install --upgrade pip && \
pip install -r requirements.txt

# Start FastAPI backend in the background
uvicorn app:api --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend
streamlit run app.py --server.port 10000



