import os
import re
import fitz
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import base64
from fpdf import FPDF

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

RED_FLAGS = [
    r"at-will", r"non[- ]?compete", r"termination.*discretion", r"subject to change",
    r"binding arbitration", r"waive.*rights", r"no liability", r"without notice", r"not obligated"
]

# --- Set Compact Mode & Query Params ---
st.set_page_config(page_title="Docudant Embedded", layout="wide")
params = st.query_params
doc_type = params.get("docType", ["Contract"])[0]
model_choice = params.get("model", ["gpt-4"])[0]
compact_mode = params.get("compactMode", ["false"])[0].lower() == "true"

components.html("""<script async defer data-domain="docudant.com" src="https://plausible.io/js/script.js"></script>""", height=0)

if not compact_mode:
    st.markdown(f"<h2 style='color:#003366'>📄 Embedded Docudant – {doc_type}</h2>", unsafe_allow_html=True)
    st.markdown("Upload your PDF to get a smart summary and risk analysis.")

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# --- Util Functions (reuse from main app) ---
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
        return f"[OCR Error: {e}]"
    return text

def ask_gpt(prompt, model="gpt-4", temperature=0.4):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def highlight_red_flags(text):
    highlighted = text
    for pattern in RED_FLAGS:
        highlighted = re.sub(
            pattern,
            lambda m: f"<span style='color:red; font-weight:bold;'>{m.group(0)}</span>",
            highlighted,
            flags=re.IGNORECASE
        )
    return highlighted

def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_line)
    pdf.output(filename, 'F')

# --- Main Flow ---
if uploaded_file:
    components.html("<script>plausible('file_uploaded')</script>", height=0)
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))
    if not text.strip():
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))

    if not text or text.strip().startswith("[OCR Error:"):
        st.error("No readable text could be extracted from this PDF.")
    else:
        st.markdown("### 🔍 Document Preview (Red = flagged)")
        highlighted = highlight_red_flags(text)
        st.markdown(f"<div style='white-space: pre-wrap'>{highlighted}</div>", unsafe_allow_html=True)

        st.markdown("### 📋 AI Summary")
        sections = {
            "Key Clauses": f"Extract the key clauses from this {doc_type}.",
            "Risks Identified": f"What are potential risks or vague/missing terms in this {doc_type}?",
            "Plain English Summary": f"Explain this {doc_type} in plain English.",
        }

        compiled = f"DOCUMENT TYPE: {doc_type}\nMODEL USED: {model_choice}\n\n"
        for title, prompt in sections.items():
            st.subheader(title)
            result = ask_gpt(prompt + "\n\n" + text, model=model_choice)
            st.text_area(title, result, height=200)
            compiled += f"--- {title.upper()} ---\n{result}\n\n"

        save_as_pdf(compiled, "embedded_summary.pdf")
        with open("embedded_summary.pdf", "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="embedded_summary.pdf">📥 Download Summary</a>'
            st.markdown(href, unsafe_allow_html=True)

# --- Legal Footer ---
if not compact_mode:
    st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")
