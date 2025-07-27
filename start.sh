#!/bin/bash

# ---------------------- System & Python Setup ----------------------
apt-get update && \
apt-get install -y tesseract-ocr && \
pip install --upgrade pip && \
pip install -r requirements.txt

# ---------------------- Launch Streamlit ----------------------
streamlit run streamlit_app.py --server.port 10000
