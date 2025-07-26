from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import shutil
import os
import subprocess

app = FastAPI()

# ---------------------- CORS CONFIG ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with ["https://docudant.com"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- ROUTES ----------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Docudant Backend is Live</h2>
    <p>Use /upload to submit a document for analysis.</p>
    """

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    file_location = f"./uploads/{filename}"
    os.makedirs("uploads", exist_ok=True)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Optional: redirect to the Streamlit UI
    return RedirectResponse(url="/streamlit", status_code=302)

# ---------------------- STREAMLIT LAUNCH ----------------------
@app.get("/streamlit")
def launch_streamlit():
    # Launch Streamlit as a subprocess
    subprocess.Popen(["streamlit", "run", "streamlit_app.py"])
    return HTMLResponse(content="Launching Streamlit... Please wait and refresh in a moment.", status_code=200)
