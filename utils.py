import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
from openai import OpenAI
from fpdf import FPDF
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Extract text from a standard PDF
def extract_text_from_pdf(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# OCR fallback using PyMuPDF and pytesseract
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

# Call OpenAI's GPT model with a prompt
def ask_gpt(prompt, model="gpt-4", temperature=0.4):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# Save summary output to a downloadable PDF
def save_as_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
