# ✅ Docudant – Final Streamlit App (with model fallback and all core enhancements)

import os
import re
import fitz
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI, AuthenticationError, OpenAIError
from dotenv import load_dotenv
import base64
from fpdf import FPDF

# --- Initialization ---
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

RED_FLAGS = [
    r"at-will", r"non[- ]?compete", r"termination.*discretion", r"subject to change",
    r"binding arbitration", r"waive.*rights", r"no liability", r"without notice", r"not obligated"
]

# --- Page Config & Analytics ---
st.set_page_config(page_title="Docudant – Contract & Offer Review AI", layout="wide")
components.html("""<script async defer data-domain="docudant.com" src="https://plausible.io/js/script.js"></script>""", height=0)

# --- Header ---
st.markdown("""
<h1 style='font-size: 2.5em; color: #003366;'>📄 Docudant – Contract & Offer Review AI</h1>
<p style='font-size: 1.1em;'>Analyze contracts, offer letters, NDAs, leases & more – with instant AI insights.</p>
<p><b>Upload a supported document for AI review. Outputs are saved locally.</b></p>
""", unsafe_allow_html=True)

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_context" not in st.session_state:
    st.session_state.doc_context = None

# --- Model and Document Type ---
model_choice = st.radio("Select model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
ALLOWED_MODELS = ["gpt-4", "gpt-3.5-turbo"]
if model_choice not in ALLOWED_MODELS:
    st.error("⚠️ Invalid model selected. Please choose a supported model.")
    st.stop()

document_type = st.selectbox("Select document type:", [
    "Contract", "Offer Letter", "Employment Agreement", "NDA", "Equity Grant", "Lease Agreement",
    "MSA", "Freelance / Custom Agreement", "Insurance Document", "Healthcare Agreement"
])
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# --- Utilities ---
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

def ask_gpt(prompt, model="gpt-4", temperature=0.4):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except (OpenAIError, AuthenticationError, ValueError) as e:
        if model == "gpt-4":
            try:
                fallback_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return "(Fallback to GPT-3.5)\n" + fallback_response.choices[0].message.content.strip()
            except Exception as fallback_error:
                return f"⚠️ GPT Error (Fallback failed): {fallback_error}"
        return f"⚠️ GPT Error: {e}"

def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe_line)
    pdf.output(filename, 'F')

# --- Document Comparison ---
st.markdown("---")
st.subheader("📄 Compare Two Documents")
doc1 = st.file_uploader("Upload first document", type=["pdf"], key="compare1")
doc2 = st.file_uploader("Upload second document", type=["pdf"], key="compare2")

if doc1 and doc2:
    text1 = extract_text_from_pdf(doc1)
    text2 = extract_text_from_pdf(doc2)
    if not text1.strip(): text1 = ocr_pdf_with_pymupdf(BytesIO(doc1.read()))
    if not text2.strip(): text2 = ocr_pdf_with_pymupdf(BytesIO(doc2.read()))
    compare_prompt = f"Compare the following two {document_type} documents and summarize the differences.\n\nDocument A:\n{text1}\n\nDocument B:\n{text2}"
    comparison_result = ask_gpt(compare_prompt, model=model_choice)
    st.text_area("Comparison Summary", comparison_result, height=300)
    components.html("<script>plausible('doc_comparison')</script>", height=0)

# --- Main Flow ---
if uploaded_file:
    st.success("✅ File uploaded successfully.")
    components.html("<script>plausible('file_uploaded')</script>", height=0)
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
        st.markdown(f"<div style='white-space: pre-wrap'>{highlight_red_flags(text)}</div>", unsafe_allow_html=True)

        st.subheader("🔺 Red Flags & Follow-Ups")
        for pattern in RED_FLAGS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                st.markdown(f"❗ **{match}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Explain: {match}", key=f"explain_{match}"):
                        st.info(ask_gpt(f"Explain why this could be a red flag: '{match}'", model=model_choice))
                with col2:
                    if st.button(f"💡 Negotiate: {match}", key=f"negotiate_{match}"):
                        st.success(ask_gpt(f"How might someone negotiate or improve this clause: '{match}'", model=model_choice))

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
            result = ask_gpt(prompt + "\n\n" + text, model=model_choice)
            st.text_area(section, result, height=300)
            output_sections[section] = result

        st.subheader("Custom Question")
        user_q = st.text_input("Ask a question about the document")
        if user_q:
            answer = ask_gpt(f"Document type: {document_type}\n\nDocument:\n{text}\n\nQuestion: {user_q}", model=model_choice)
            st.text_area("Answer", answer, height=200)
            components.html("<script>plausible('custom_question')</script>", height=0)

        compiled_context = f"Document Type: {document_type}\n\nExtracted Summary:\n\n"
        for title, content in output_sections.items():
            compiled_context += f"--- {title.upper()} ---\n{content}\n\n"
        st.session_state.doc_context = compiled_context

        save_as_pdf(compiled_context, "summary.pdf")
        with open("summary.pdf", "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="summary.pdf">📅 Download PDF Summary</a>'
            st.markdown(href, unsafe_allow_html=True)
            components.html("<script>plausible('download_summary')</script>", height=0)

        st.session_state.history.append({
            "type": document_type,
            "text": text,
            "results": output_sections
        })

# --- Saved Summaries Viewer ---
st.markdown("---")
if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### 📄 Saved Summary {len(st.session_state.history) - i}")
            st.markdown(f"**Type:** {entry['type']}")
            for title, content in entry["results"].items():
                with st.expander(title):
                    st.markdown(content)

# --- Conversational Q&A ---
st.markdown("---")
st.markdown("## 💬 Ask Docudant About Your Document")

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Ask a question about the uploaded document...")
if user_input:
    if st.session_state.doc_context:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt = f"You are a document review assistant. A user has uploaded a {document_type}. The summary of the document is:\n\n{st.session_state.doc_context}\n\nThe user is asking: {user_input}"
                reply = ask_gpt(prompt, model=model_choice)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.markdown(reply)
    else:
        st.warning("Please upload and process a document first to enable contextual Q&A.")

# --- Disclaimer ---
st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")
