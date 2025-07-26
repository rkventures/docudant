#!/bin/bash
apt-get update && \
apt-get install -y tesseract-ocr && \
pip install --upgrade pip && \
pip install -r requirements.txt

# ✅ Start the correct Streamlit app
streamlit run streamlit_app.py --server.port 10000
