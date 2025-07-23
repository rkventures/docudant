import os
import re
import fitz
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import streamlit as st
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import base64
from fpdf import FPDF
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------------------- FastAPI Backend ----------------------
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}

# ---------------------- Load Env and Init ----------------------
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Docudant – Contract & Offer Review AI", layout="wide")
st.title("📄 Docudant – Contract & Offer Review AI")
st.markdown("_Analyze contracts, offer letters, NDAs, leases & more – with instant AI insights._")
st.markdown("Upload a supported document for AI review. Outputs are saved locally.")

if "history" not in st.session_state:
    st.session_state.history = []

model_choice = st.radio("Select model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True, key="model_choice_main")

document_type = st.selectbox("Select document type:", [
    "Contract", "Offer Letter", "Employment Agreement", "NDA",
    "Equity Grant", "Lease Agreement", "MSA", "Freelance / Custom Agreement"
])

st.markdown("""ℹ️ **Examples of documents this tool can analyze:**
- Freelance and consulting contracts  
- Startup equity offers and RSU agreements  
- Lease and rental agreements  
- Vendor NDAs and MSAs  
- Custom one-off agreements  
""")

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"], label_visibility="collapsed", key="main_upload")

# ---------------------- Text Extraction ----------------------
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def ocr_pdf_with_pymupdf(file):
    text = ""
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            blocks = page.get_text("blocks")
            if blocks and any(block[4].strip() for block in blocks):
                text += page.get_text()
            else:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img)
    except Exception as e:
        st.error(f"OCR failed: {e}")
        return f"[OCR Error: {e}]"
    return text

# ---------------------- GPT Prompting ----------------------
def ask_gpt(prompt, model="gpt-4", temperature=0.4):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except AuthenticationError as e:
        return f"⚠️ Error: {e}"

# ---------------------- Save as PDF ----------------------
def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_line)
    pdf.output(filename, 'F')

# ---------------------- Main Analyzer ----------------------
if uploaded_file:
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))
    if not text.strip():
        st.warning("PDF appears to contain no extractable text. Attempting OCR fallback...")
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))

    if not text or text.strip().startswith("[OCR Error:"):
        st.error("❌ No readable text could be extracted from this PDF.")
    else:
        st.markdown("### 📄 Document Preview")
        st.text_area("Preview", text, height=300)

        sections = {
            "Parties & Roles": f"In this {document_type}, who are the involved parties and what are their roles? Provide in plain English.",
            "Key Clauses": f"Extract the key clauses from this {document_type}. Summarize each clause clearly.",
            "Plain English Explanations": f"Explain each clause from this {document_type} in simple, plain English.",
            "Risks Identified": f"Identify any vague, risky, or missing terms in this {document_type}. Explain why they're risky or vague.",
            "Negotiation Suggestions": f"What are potential negotiation points or improvement suggestions for the terms in this {document_type}?",
            "Clause Benchmarking": f"Compare the key clauses in this {document_type} to standard industry benchmarks. Highlight deviations.",
            "Clause Suggestions": f"Suggest missing clauses or commonly expected provisions for this type of {document_type} that are not included.",
            "Smart Next Steps": f"Based on this {document_type}, suggest smart next steps or actions a professional should consider."
        }

        output_sections = {}

        for section, prompt in sections.items():
            st.subheader(section)
            response = ask_gpt(prompt + "\n\n" + text, model=model_choice)
            st.text_area(section, response, height=300)
            output_sections[section] = response

        # Custom Question
        st.subheader("Custom Question")
        user_q = st.text_input("Ask a question about the document", key="custom_question")
        if user_q:
            answer = ask_gpt(f"Document type: {document_type}\n\nDocument:\n{text}\n\nQuestion: {user_q}", model=model_choice)
            st.text_area("Answer", answer, height=200)

        # Save PDF Summary
        compiled = f"""DOCUMENT TYPE: {document_type}\nMODEL USED: {model_choice}\n\n"""
        for title, content in output_sections.items():
            compiled += f"--- {title.upper()} ---\n{content}\n\n"

        filename = "document_review_summary.pdf"
        save_as_pdf(compiled, filename)

        with open(filename, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download PDF Summary</a>'
            st.markdown(href, unsafe_allow_html=True)

        st.session_state.history.append({
            "type": document_type,
            "text": text,
            "results": output_sections
        })

# ---------------------- Feedback and History ----------------------
st.markdown("💬 Found something unclear or helpful? [Click here to leave feedback](https://forms.gle/yourformlink)")
st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")

if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### 📄 Saved Summary {len(st.session_state.history) - i}")
            st.markdown(f"**Type:** {entry['type']}")
            for title, content in entry["results"].items():
                with st.expander(title):
                    st.markdown(content)

# ---------------------- Document Comparison ----------------------
st.markdown("---")
st.header("🆚 Compare Document Versions (Optional)")
model_choice_2 = st.radio("Select model for comparison", ["gpt-4", "gpt-3.5-turbo"], horizontal=True, key="model_choice_compare")
old_doc = st.file_uploader("Upload previous version", type=["pdf"], key="old_upload")
new_doc = st.file_uploader("Upload current version", type=["pdf"], key="new_upload")

if old_doc and new_doc:
    old_text = extract_text_from_pdf(old_doc)
    if not old_text.strip():
        old_text = ocr_pdf_with_pymupdf(old_doc)

    new_text = extract_text_from_pdf(new_doc)
    if not new_text.strip():
        new_text = ocr_pdf_with_pymupdf(new_doc)

    diff_prompt = f"Compare these two versions of a {document_type} and highlight all meaningful differences in clauses, responsibilities, compensation, and obligations.\n\n--- OLD VERSION ---\n{old_text}\n\n--- NEW VERSION ---\n{new_text}"
    comparison = ask_gpt(diff_prompt, model=model_choice_2)
    st.subheader("Comparison Result")
    st.text_area("Differences", comparison, height=300)
