#!/bin/bash
apt-get update && \
apt-get install -y tesseract-ocr && \
pip install --upgrade pip && \
pip install -r requirements.txt

# Start FastAPI or Streamlit based on your app
streamlit run app.py


