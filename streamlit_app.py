# ✅ Docudant – Final Streamlit App (All MVP Features Restored)
# Features: OCR fallback, GPT fallback, red flag highlighting, smart next steps,
# clause benchmarking, compensation estimation, compensation benchmarking,
# document comparison, saved summaries, and contextual Q&A

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
import difflib

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

RED_FLAGS = [
    r"at-will", r"non[- ]?compete", r"termination.*discretion", r"subject to change",
    r"binding arbitration", r"waive.*rights", r"no liability", r"without notice", r"not obligated"
]

st.set_page_config(page_title="Docudant – Contract & Offer Review AI", layout="wide")
components.html("""<script async defer data-domain="docudant.com" src="https://plausible.io/js/script.js"></script>""", height=0)

st.markdown("""
<h1 style='font-size: 2.5em; color: #003366;'>📄 Docudant – Contract & Offer Review AI</h1>
<p style='font-size: 1.1em;'>Analyze contracts, offer letters, NDAs, leases & more – with instant AI insights.</p>
<p><b>Upload a supported document for AI review. Outputs are saved locally.</b></p>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_context" not in st.session_state:
    st.session_state.doc_context = None
if "previous_doc" not in st.session_state:
    st.session_state.previous_doc = None

model_choice = st.radio("Select model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
document_type = st.selectbox("Select document type:", [
    "Contract", "Offer Letter", "Employment Agreement", "NDA", "Equity Grant", "Lease Agreement",
    "MSA", "Freelance / Custom Agreement", "Insurance Document", "Healthcare Agreement"
])
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
ALLOWED_MODELS = ["gpt-4", "gpt-3.5-turbo"]
if model_choice not in ALLOWED_MODELS:
    st.error("⚠️ Invalid model selected.")
    st.stop()

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
    for pattern in RED_FLAGS:
        text = re.sub(pattern, lambda m: f"<span style='color:red; font-weight:bold;'>{m.group(0)}</span>", text, flags=re.IGNORECASE)
    return text

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
                fallback = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
                )
                return "(Fallback to GPT-3.5)\n" + fallback.choices[0].message.content.strip()
            except Exception as fallback_error:
                return f"⚠️ GPT Error (Fallback failed): {fallback_error}"
        return f"⚠️ GPT Error: {e}"

def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        safe = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe)
    pdf.output(filename, 'F')

def estimate_offer_compensation(text):
    prompt = f"""You're an expert HR and compensation analyst. Extract and summarize all components of total compensation in this offer letter...

{text}"""
    return ask_gpt(prompt)

def benchmark_offer_compensation(text):
    prompt = f"""You are a compensation benchmarking expert. Based on the offer letter below, analyze whether the compensation offered is competitive...

{text}"""
    return ask_gpt(prompt)

def extract_clauses(text):
    pattern = r"(?:(?:^|\n)(?:\d+\.|\d+\)|[A-Z]\.)\s+.+?)(?=\n\d+\.|\n\d+\)|\n[A-Z]\.|\Z)"
    return [c.strip() for c in re.findall(pattern, text, re.DOTALL) if len(c.strip()) > 30]

def benchmark_clause_against_industry(clause, doc_type):
    prompt = f"You are a contract review expert. Benchmark the following clause from a {doc_type} against industry standards:\n\n\"\"\"{clause}\"\"\""
    return ask_gpt(prompt)

def compare_documents(old_text, new_text):
    diff = difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm='')
    return '\n'.join(diff)

# --- Main Flow ---
if uploaded_file:
    st.success("✅ File uploaded.")
    components.html("<script>plausible('file_uploaded')</script>", height=0)
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))
    if not text.strip():
        st.warning("PDF has no text. Trying OCR...")
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))
    if not text or text.strip().startswith("[OCR Error:"):
        st.error("❌ Text could not be extracted.")
    else:
        st.text_area("🔍 Text Preview", text[:1000])
        st.markdown("### 📄 Document (Red Flags Highlighted)")
        st.markdown(f"<div style='white-space: pre-wrap'>{highlight_red_flags(text)}</div>", unsafe_allow_html=True)

        if document_type == "Offer Letter":
            st.subheader("💰 Compensation Breakdown")
            st.text_area("Estimated Compensation", estimate_offer_compensation(text), height=250)
            st.subheader("📊 Compensation Benchmark")
            st.text_area("Benchmark Insights", benchmark_offer_compensation(text), height=250)

        st.subheader("🔺 Red Flags & Follow-Ups")
        for pattern in RED_FLAGS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                st.markdown(f"❗ **{match}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Explain: {match}", key=f"explain_{match}"):
                        st.info(ask_gpt(f"Why is this a red flag: '{match}'?", model=model_choice))
                with col2:
                    if st.button(f"Negotiate: {match}", key=f"negotiate_{match}"):
                        st.success(ask_gpt(f"How to negotiate or improve: '{match}'?", model=model_choice))

        sections = {
            "Parties & Roles": f"In this {document_type}, who are the involved parties and their roles?",
            "Key Clauses": f"Extract the key clauses from this {document_type}.",
            "Plain English Explanations": f"Explain each clause in plain English.",
            "Risks Identified": f"What are potential risks or vague terms?",
            "Negotiation Suggestions": f"What should a person negotiate in this {document_type}?",
            "Clause Suggestions": f"Suggest any missing clauses.",
            "Smart Next Steps": f"What smart actions should be taken next?"
        }

        summary = {}
        for title, prompt in sections.items():
            st.subheader(title)
            result = ask_gpt(prompt + "\n\n" + text, model=model_choice)
            st.text_area(title, result, height=300)
            summary[title] = result

        st.subheader("📊 Clause Benchmarking")
        for i, clause in enumerate(extract_clauses(text)[:10]):
            with st.expander(f"Clause {i+1}"):
                st.markdown(f"**Clause Text:**\n\n{clause}")
                st.markdown(f"**Feedback:**\n\n{benchmark_clause_against_industry(clause, document_type)}")

        if st.session_state.previous_doc:
            st.subheader("📑 Document Comparison (Previous vs Current)")
            diff = compare_documents(st.session_state.previous_doc, text)
            st.text_area("Differences", diff if diff else "✅ No differences found", height=300)

        st.session_state.previous_doc = text

        st.subheader("Ask a Custom Question")
        custom_q = st.text_input("Question")
        if custom_q:
            reply = ask_gpt(f"Document type: {document_type}\n\n{text}\n\nQ: {custom_q}", model=model_choice)
            st.text_area("Answer", reply, height=200)

        doc_summary = f"Document Type: {document_type}\n\n"
        for t, c in summary.items():
            doc_summary += f"--- {t.upper()} ---\n{c}\n\n"
        st.session_state.doc_context = doc_summary

        save_as_pdf(doc_summary, "summary.pdf")
        with open("summary.pdf", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            link = f'<a href="data:application/octet-stream;base64,{b64}" download="summary.pdf">📄 Download PDF Summary</a>'
            st.markdown(link, unsafe_allow_html=True)

        st.session_state.history.append({
            "type": document_type,
            "text": text,
            "results": summary
        })

# --- Saved Summaries Viewer ---
if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### Summary {len(st.session_state.history)-i} – {entry['type']}")
            for t, c in entry["results"].items():
                with st.expander(t):
                    st.markdown(c)

# --- Conversational Q&A ---
st.markdown("## 💬 Ask Docudant About Your Document")
for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).markdown(msg["content"])
user_input = st.chat_input("Ask a question...")
if user_input and st.session_state.doc_context:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = f"The user uploaded a {document_type}. Summary:\n\n{st.session_state.doc_context}\n\nQuestion: {user_input}"
            answer = ask_gpt(context, model=model_choice)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.markdown(answer)

# --- Disclaimer ---
st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")
