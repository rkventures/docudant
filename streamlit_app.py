# ✅ Docudant: streamlit_app.py (Updated with working Plausible Analytics)
# ---------------------------------------------------
# ✅ Feature Checklist (7/27/2025)
# [x] GPT Model selector (gpt-4 / gpt-3.5-turbo)
# [x] Document type selector (Offer, Contract, NDA, etc.)
# [x] PDF text extraction with OCR fallback
# [x] Red flag highlighting
# [x] GPT section-based analysis
# [x] Smart Next Steps
# [x] Custom question prompt
# [x] PDF summary download
# [x] Saved history viewer
# [x] Document comparison engine ✅
# [x] Removed debug/prompts from output ✅
# [x] Unicode error fix for emoji surrogates ✅
# [x] ✅ Plausible Analytics FIXED

import os
import re
import fitz
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI, AuthenticationError
from dotenv import load_dotenv
import base64
from fpdf import FPDF

# ---------------------- Load Env and Init ----------------------
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ---------------------- Red Flag Patterns ----------------------
RED_FLAGS = [
    r"at-will",
    r"non[- ]?compete",
    r"termination.*discretion",
    r"subject to change",
    r"binding arbitration",
    r"waive.*rights",
    r"no liability",
    r"without notice",
    r"not obligated"
]

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Docudant – Contract & Offer Review AI", layout="wide")

# ✅ Working Plausible Analytics Snippet
components.html("""
<script async defer data-domain="docudant.com" src="https://plausible.io/js/script.js"></script>
""", height=0)

st.title("📄 Docudant – Contract & Offer Review AI")
st.markdown("_Analyze contracts, offer letters, NDAs, leases & more – with instant AI insights._")
st.markdown("Upload a supported document for AI review. Outputs are saved locally.")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------- Upload UI ----------------------
model_choice = st.radio("Select model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
document_type = st.selectbox("Select document type:", [
    "Contract", "Offer Letter", "Employment Agreement", "NDA",
    "Equity Grant", "Lease Agreement", "MSA", "Freelance / Custom Agreement",
    "Insurance Document", "Healthcare Agreement"
], index=0)

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

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
        return f"[OCR Error: {e}]"
    return text

# ---------------------- Red Flag Highlighting ----------------------
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

# ---------------------- PDF Saving ----------------------
def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_line)
    pdf.output(filename, 'F')

# ---------------------- Comparison ----------------------
st.markdown("---")
st.subheader("📄 Compare Two Documents")
doc1 = st.file_uploader("Upload first document", type=["pdf"], key="compare1")
doc2 = st.file_uploader("Upload second document", type=["pdf"], key="compare2")

if doc1 and doc2:
    text1 = extract_text_from_pdf(doc1)
    text2 = extract_text_from_pdf(doc2)

    if not text1.strip():
        text1 = ocr_pdf_with_pymupdf(BytesIO(doc1.read()))
    if not text2.strip():
        text2 = ocr_pdf_with_pymupdf(BytesIO(doc2.read()))

    compare_prompt = f"Compare the following two {document_type} documents and summarize the differences.\n\nDocument A:\n{text1}\n\nDocument B:\n{text2}"
    comparison_result = ask_gpt(compare_prompt, model=model_choice)
    st.text_area("Comparison Summary", comparison_result, height=300)

# ---------------------- Main Analysis ----------------------
if uploaded_file:
    st.success("✅ File uploaded successfully.")
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))

    if not text.strip():
        st.warning("PDF appears to contain no extractable text. Attempting OCR fallback...")
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))

    if not text or text.strip().startswith("[OCR Error:"):
        st.error("❌ No readable text could be extracted from this PDF.")
    else:
        st.markdown("### 🔍 Extracted Text Preview")
        st.text_area("Preview", text[:1000])

        st.markdown("### 📄 Document Preview (Red = flagged)")
        highlighted = highlight_red_flags(text)
        st.markdown(f"<div style='white-space: pre-wrap'>{highlighted}</div>", unsafe_allow_html=True)

        sections = {
            "Parties & Roles": f"In this {document_type}, who are the involved parties and what are their roles?",
            "Key Clauses": f"Extract the key clauses from this {document_type}.",
            "Plain English Explanations": f"Explain each clause in plain English.",
            "Risks Identified": f"What are potential risks or vague/missing terms in this {document_type}?",
            "Negotiation Suggestions": f"What should a professional negotiate or ask for in this {document_type}?",
            "Clause Benchmarking": f"Compare clauses to industry benchmarks.",
            "Clause Suggestions": f"Suggest any missing clauses for a typical {document_type}.",
            "Smart Next Steps": f"Based on this {document_type}, suggest smart next steps."
        }

        output_sections = {}
        for section, prompt in sections.items():
            st.subheader(section)
            response = ask_gpt(prompt + "\n\n" + text, model=model_choice)
            st.text_area(section, response, height=300)
            output_sections[section] = response

        st.subheader("Custom Question")
        user_q = st.text_input("Ask a question about the document")
        if user_q:
            answer = ask_gpt(f"Document type: {document_type}\n\nDocument:\n{text}\n\nQuestion: {user_q}", model=model_choice)
            st.text_area("Answer", answer, height=200)

        compiled = f"DOCUMENT TYPE: {document_type}\nMODEL USED: {model_choice}\n\n"
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

# ---------------------- History ----------------------
st.markdown("---")
if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### 📄 Saved Summary {len(st.session_state.history) - i}")
            st.markdown(f"**Type:** {entry['type']}")
            for title, content in entry["results"].items():
                with st.expander(title):
                    st.markdown(content)

st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")
