# ✅ Docudant – Final Streamlit App (All MVP Features Restored)
# Features: OCR fallback, GPT fallback, red flag highlighting, smart next steps,
# clause benchmarking, compensation estimation, compensation benchmarking,
# document comparison, saved summaries, and contextual Q&A
# ✅ PDF Summary includes compensation + clause benchmarking + fallback labels

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

# [...] OMITTED FOR BREVITY — full updated code will continue below this in a second chunk


# --- State Setup ---
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
debug_mode = st.sidebar.checkbox("🔧 Enable Debug Mode")
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
    
    except (OpenAIError, AuthenticationError, ValueError, Exception) as e:
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
    pdf.set_font("Arial", size=11)  # ✅ Consistent font
    pdf.set_auto_page_break(auto=True, margin=15)
    for line in text.split('\n'):
        safe = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, safe)
    pdf.output(filename, 'F')



def estimate_offer_compensation(text):
    prompt = f"""You're an expert HR and compensation analyst. Extract and summarize all components of total compensation in this offer letter:

{text}"""
    return ask_gpt(prompt, model=model_choice)


def benchmark_offer_compensation(text):
    prompt = f"""You are a compensation benchmarking expert. Based on the offer letter below, analyze whether the compensation offered is competitive:

{text}"""
    return ask_gpt(prompt, model=model_choice)


def extract_clauses(text):
    # ✅ FIXED: Pattern indentation and newlines for bullet/numbered clauses
    pattern = r"(?:(?:^|\n)(?:\d+\.|\d+\)|[A-Z]\.)\s+.+?)(?=\n\d+\.|\n\d+\)|\n[A-Z]\.|\Z)"
    return [c.strip() for c in re.findall(pattern, text, re.DOTALL) if len(c.strip()) > 30]


def benchmark_clause_against_industry(clause, doc_type):
    # ✅ FIXED: triple quotes were unbalanced
    prompt = f"""Benchmark the following clause from a {doc_type} against industry standards:

\"\"\"{clause}\"\"\""""
    return ask_gpt(prompt, model=model_choice)


def compare_documents(old_text, new_text):
    diff = difflib.unified_diff(old_text.splitlines(), new_text.splitlines(), lineterm='')
    return '\n'.join(diff)

# --- Main Logic Flow ---
# ... [Final block of logic with all analysis sections will continue in next cell]


if uploaded_file:
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))
    if not text.strip():
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))

    if not text or text.strip().startswith("[OCR Error:"):
        st.error("❌ Text could not be extracted.")
        st.stop()

    if debug_mode:
        st.sidebar.write("✅ Text Length:", len(text))
        st.sidebar.write("✅ Extracted Clauses:", len(extract_clauses(text)))
        st.sidebar.write("✅ Red Flag Patterns Matched:", len([
            m for p in RED_FLAGS for m in re.findall(p, text, re.IGNORECASE)
        ]))

    st.text_area("🔍 Text Preview", text[:1000])

    st.markdown("### 🆚 Document Comparison (Optional)")
    comparison_file = st.file_uploader("Upload a second version to compare", type=["pdf"], key="compare")
    if comparison_file:
        compare_text = extract_text_from_pdf(comparison_file)
        if not compare_text.strip():
            compare_text = ocr_pdf_with_pymupdf(comparison_file)
        if compare_text:
            prompt = f"Compare these two versions of a {document_type}:\n\n--- A ---\n{text}\n\n--- B ---\n{compare_text}"
            diff_result = ask_gpt(prompt, model=model_choice)
            st.text_area("Comparison Summary", diff_result, height=400)

    st.markdown("### 📄 Red Flags")
    st.markdown("**Flagged Clauses:**")
    flagged_clauses = []
    for pattern in RED_FLAGS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            snippet = text[start:end].replace('\n', ' ')
            flagged_clauses.append(f"🔺 ...{snippet}...")
    for fc in flagged_clauses:
        st.markdown(f"- {fc}")

    compensation_summary = ""
    benchmark_summary = ""
    if "Offer Letter" in document_type:
        st.subheader("💰 Compensation Breakdown")
        compensation_summary = estimate_offer_compensation(text)
        if compensation_summary:
            st.text_area("Estimated Compensation", compensation_summary, height=250)
        else:
            st.warning("⚠️ Could not estimate total compensation.")

        st.subheader("📊 Compensation Benchmark")
        benchmark_summary = benchmark_offer_compensation(text)
        if benchmark_summary:
            st.text_area("Benchmark Insights", benchmark_summary, height=250)
        else:
            st.warning("⚠️ Compensation benchmarking failed.")

    st.subheader("🔺 Red Flag Explanations")
    red_flags_text = ""
    unique_flags = set()
    for pattern in RED_FLAGS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            term = match.group(0)
            if term.lower() in unique_flags:
                continue
            unique_flags.add(term.lower())
            st.markdown(f"**🔺 {term}**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Explain: {term}", key=f"explain_{term}"):
                    st.info(ask_gpt(f"Why is this a red flag: '{term}'?", model=model_choice))
            with col2:
                if st.button(f"Negotiate: {term}", key=f"negotiate_{term}"):
                    st.success(ask_gpt(f"How to negotiate or improve: '{term}'?", model=model_choice))

    sections = {
        "Parties & Roles": f"In this {document_type}, who are the involved parties and their roles?",
        "Key Clauses": f"Extract the key clauses from this {document_type}.",
        "Plain English Explanations": f"Explain each clause in plain English.",
        "Risks Identified": f"What are potential risks or vague terms?",
        "Negotiation Suggestions": f"What should a person negotiate in this {document_type}?",
        "Clause Suggestions": f"Suggest any missing clauses.",
        "Smart Next Steps": f"What smart actions should be taken next?"
    }

    st.markdown("---")
    st.subheader("📄 Document Summary Sections")

    summary = {}
    for title, prompt in sections.items():
        result = ask_gpt(prompt + "\n\n" + text, model=model_choice)
        st.text_area(title, result, height=300)
        summary[title] = result

    st.subheader("📊 Clause Benchmarking")
    clauses = extract_clauses(text)
    clause_summaries = ""

    if not clauses:
        st.warning("⚠️ No clauses extracted for benchmarking.")
    else:
        for i, clause in enumerate(clauses[:10]):
            with st.expander(f"Clause {i+1}"):
                st.markdown(f"**Clause Text:**\n\n{clause}")
                try:
                    feedback = benchmark_clause_against_industry(clause, document_type)
                    st.markdown(f"**Feedback:**\n\n{feedback}")
                    clause_summaries += f"--- CLAUSE {i+1} ---\n{feedback}\n"
                except Exception as e:
                    st.error(f"⚠️ GPT error during benchmarking: {e}")

    if st.session_state.previous_doc:
        st.subheader("📑 Previous vs Current Comparison")
        diff = compare_documents(st.session_state.previous_doc, text)
        st.text_area("Differences", diff if diff else "✅ No differences found", height=300)
    st.session_state.previous_doc = text

    st.subheader("Ask a Custom Question")
    custom_q = st.text_input("Question")
    if custom_q:
        reply = ask_gpt(f"Document type: {document_type}\n\n{text}\n\nQ: {custom_q}", model=model_choice)
        st.text_area("Answer", reply, height=200)

    # ✅ This block must be indented
    doc_summary = f"Document Type: {document_type}\n\n"
    doc_summary += f"{red_flags_text}\n"
    for t, c in summary.items():
        doc_summary += f"--- {t.upper()} ---\n{c}\n"
    doc_summary += f"--- CLAUSE BENCHMARKING ---\n{clause_summaries}\n"
    doc_summary += f"--- COMPENSATION ---\n{compensation_summary or '⚠️ No compensation data'}\n"
    doc_summary += f"--- BENCHMARK ---\n{benchmark_summary or '⚠️ No benchmark data'}\n"


    st.session_state.doc_context = doc_summary
    save_as_pdf(doc_summary, "summary.pdf")
    with open("summary.pdf", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        link = f'<a href="data:application/octet-stream;base64,{b64}" download="summary.pdf">📄 Download PDF Summary</a>'
        st.markdown(link, unsafe_allow_html=True)

    st.session_state.history.append({"type": document_type, "text": text, "results": summary})

if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### Summary {len(st.session_state.history)-i} – {entry['type']}")
            for t, c in entry["results"].items():
                with st.expander(t):
                    st.markdown(c)

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

st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")
