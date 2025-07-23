#!/bin/bash
apt-get update && \
apt-get install -y tesseract-ocr && \
pip install --upgrade pip && \
pip install -r requirements.txt

# Start only the Streamlit frontend
streamlit run app.py --server.port 10000
