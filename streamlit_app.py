# ... (all existing imports remain unchanged)
if "history" not in st.session_state:
    st.session_state.history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_context" not in st.session_state:
    st.session_state.doc_context = None

# --- Existing Page Config and Header (unchanged) ---
# ... (same st.set_page_config, plausible script, title markdown, etc.)

# --- Model & Document Type Selection ---
model_choice = st.radio("Select model", ["gpt-4", "gpt-3.5-turbo"], horizontal=True)
document_type = st.selectbox("Select document type:", [...])  # unchanged

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# --- Document Comparison Section (unchanged) ---
# ... (doc1, doc2, comparison, plausible logging)

# --- Document Upload Handling & Analysis ---
if uploaded_file:
    st.success("\u2705 File uploaded successfully.")
    components.html("<script>plausible('file_uploaded')</script>", height=0)
    file_bytes = uploaded_file.read()
    text = extract_text_from_pdf(BytesIO(file_bytes))
    if not text.strip():
        st.warning("PDF appears to contain no extractable text. Attempting OCR fallback...")
        text = ocr_pdf_with_pymupdf(BytesIO(file_bytes))
    if not text or text.strip().startswith("[OCR Error:"):
        st.error("\u274C No readable text could be extracted from this PDF.")
    else:
        st.markdown("### 🔍 Extracted Text Preview")
        st.text_area("Preview", text[:1000])

        st.markdown("### 📄 Document Preview (Red = flagged)")
        highlighted = highlight_red_flags(text)
        st.markdown(f"<div style='white-space: pre-wrap'>{highlighted}</div>", unsafe_allow_html=True)

        # 🔺 Red Flag Smart Follow-ups
        st.subheader("🔺 Red Flags & Follow-Ups")
        for pattern in RED_FLAGS:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            for match in matches:
                st.markdown(f"❗ **{match}**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Explain: {match}", key=f"explain_{match}"):
                        explanation = ask_gpt(f"Explain why this could be a red flag: '{match}'", model=model_choice)
                        st.info(explanation)
                with col2:
                    if st.button(f"💡 Negotiate: {match}", key=f"negotiate_{match}"):
                        suggestion = ask_gpt(f"How might someone negotiate or improve this clause: '{match}'", model=model_choice)
                        st.success(suggestion)

        # 🔍 Structured Section Analysis
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
            components.html("<script>plausible('custom_question')</script>", height=0)

        # Save context for Q&A
        compiled_context = f"Document Type: {document_type}\n\nExtracted Summary:\n\n"
        for title, content in output_sections.items():
            compiled_context += f"--- {title.upper()} ---\n{content}\n\n"
        st.session_state.doc_context = compiled_context

        filename = "document_review_summary.pdf"
        save_as_pdf(compiled_context, filename)

        with open(filename, "rb") as file:
            b64 = base64.b64encode(file.read()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download PDF Summary</a>'
            st.markdown(href, unsafe_allow_html=True)
            components.html("<script>plausible('download_summary')</script>", height=0)

        st.session_state.history.append({
            "type": document_type,
            "text": text,
            "results": output_sections
        })

# --- Saved Summaries Viewer (unchanged) ---
st.markdown("---")
if st.session_state.history:
    with st.expander("📚 View Saved Summaries"):
        for i, entry in enumerate(reversed(st.session_state.history[-3:])):
            st.markdown(f"### 📄 Saved Summary {len(st.session_state.history) - i}")
            st.markdown(f"**Type:** {entry['type']}")
            for title, content in entry["results"].items():
                with st.expander(title):
                    st.markdown(content)

# --- 💬 Conversational Q&A UI ---
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
                prompt = f"""You are a document review assistant. A user has uploaded a {document_type}. The summary of the document is:\n\n{st.session_state.doc_context}\n\nThe user is asking: {user_input}"""
                reply = ask_gpt(prompt, model=model_choice)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.markdown(reply)
    else:
        st.warning("Please upload and process a document first to enable contextual Q&A.")

# --- Legal Disclaimer ---
st.markdown("---\n_Disclaimer: This summary is AI-generated and should not be considered legal advice._")
