import os
import re
import fitz
import pytesseract
import streamlit as st
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import base64
from fpdf import FPDF

# Load environment and OpenAI client
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# Streamlit setup
st.set_page_config(page_title="AI Document Review Agent (Final)", layout="wide")
st.title("📄 AI Document Review Agent (Final)")
st.markdown("Upload a supported document for AI review. Outputs are saved locally.")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Model selector
model_choice = st.radio("Select model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)

# Document type selector
document_type = st.selectbox("Select document type:", [
    "Contract", "Offer Letter", "Employment Agreement", "NDA",
    "Equity Grant", "Lease Agreement", "MSA", "Freelance / Custom Agreement"
])

st.markdown("""
ℹ️ **Examples of documents this tool can analyze:**
- Freelance and consulting contracts  
- Startup equity offers and RSU agreements  
- Lease and rental agreements  
- Vendor NDAs and MSAs  
- Custom one-off agreements  
""")

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"], label_visibility="collapsed")

# Text extraction functions
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
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        blocks = page.get_text("blocks")
        if blocks and any(block[4].strip() for block in blocks):
            text += page.get_text()
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.open(BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img)
    return text

# GPT prompt helper
def ask_gpt(prompt, model="gpt-4", temperature=0.4):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# Save as PDF
def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

# Document analyzer
if uploaded_file:
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))
    if not text.strip():
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))

    if not text:
        st.error("No text could be extracted from the uploaded PDF.")
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

        # Custom question input
        st.subheader("Custom Question")
        user_q = st.text_input("Ask a question about the document")
        if user_q:
            answer = ask_gpt(f"Document type: {document_type}\n\nDocument:\n{text}\n\nQuestion: {user_q}", model=model_choice)
            st.text_area("Answer", answer, height=200)

        # Save everything as PDF
        compiled = f"""DOCUMENT TYPE: {document_type}\nMODEL USED: {model_choice}\n\n"""
        for title, content in output_sections.items():
            compiled += f"--- {title.upper()} ---\n{content}\n\n"

        filename = "document_review_summary.pdf"
        save_as_pdf(compiled, filename)

        with open(filename, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Summary</a>'
            st.markdown(href, unsafe_allow_html=True)

        # Add to history
        st.session_state.history.append({
            "type": document_type,
            "text": text,
            "results": output_sections
        })

# Feedback and disclaimer
st.markdown("💬 Found something unclear or helpful? [Click here to leave feedback](https://forms.gle/yourformlink)")
st.markdown("\n---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")

# Saved summaries viewer
if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### 📄 Saved Summary {len(st.session_state.history) - i}")
            st.markdown(f"**Type:** {entry['type']}")
            for title, content in entry["results"].items():
                with st.expander(title):
                    st.markdown(content)

# Compare documents
st.markdown("---")
st.header("🆚 Compare Document Versions (Optional)")
old_doc = st.file_uploader("Upload previous version", type=["pdf"], key="old")
new_doc = st.file_uploader("Upload current version", type=["pdf"], key="new")

if old_doc and new_doc:
    old_text = extract_text_from_pdf(old_doc)
    if not old_text.strip():
        old_text = ocr_pdf_with_pymupdf(old_doc)

    new_text = extract_text_from_pdf(new_doc)
    if not new_text.strip():
        new_text = ocr_pdf_with_pymupdf(new_doc)

    diff_prompt = f"Compare these two versions of a {document_type} and highlight all meaningful differences in clauses, responsibilities, compensation, and obligations.\n\n--- OLD VERSION ---\n{old_text}\n\n--- NEW VERSION ---\n{new_text}"
    comparison = ask_gpt(diff_prompt, model=model_choice)
    st.subheader("Comparison Result")
    st.text_area("Differences", comparison, height=300)
