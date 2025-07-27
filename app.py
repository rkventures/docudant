from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
import shutil
import os

app = FastAPI()

# ---------------------- CORS CONFIG ----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://docudant.com"] before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- HOME ROUTE ----------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Docudant Backend is Live</h2>
    <p>Use /upload to submit a document for analysis.</p>
    """

# ---------------------- UPLOAD ROUTE ----------------------
@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: str = Form(...)
):
    os.makedirs("uploads", exist_ok=True)

    # Basic file extension validation
    allowed_extensions = [".pdf", ".doc", ".docx"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return HTMLResponse(content="❌ Unsupported file type. Please upload PDF, DOC, or DOCX.", status_code=400)

    # Save uploaded file
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save selected document type
    with open("./uploads/last_doc_type.txt", "w") as f:
        f.write(doc_type)

    # Save the filename for Streamlit to pick up
    with open("./uploads/last_uploaded.txt", "w") as f:
        f.write(file.filename)

    # Redirect to Streamlit interface
    return RedirectResponse(url="/streamlit", status_code=302)

# ---------------------- STREAMLIT STATUS ----------------------
@app.get("/streamlit")
def launch_streamlit():
    return HTMLResponse(content="✅ Document uploaded. Please switch to the Streamlit interface.", status_code=200)
