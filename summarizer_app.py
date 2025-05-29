import os
import nltk
import PyPDF2
import streamlit as st
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === Setup Streamlit ===
st.set_page_config(page_title="📄 PDF Summarizer", layout="wide")
st.title("📘 Summarize PDF Using Local LLM")

# === Select model ===
model_options = {
    "BART (facebook/bart-large-cnn)": "facebook/bart-large-cnn",
    "T5 (t5-base)": "t5-base",
    "Pegasus (google/pegasus-xsum)": "google/pegasus-xsum"
}
selected_model_label = st.selectbox("🔁 Choose a summarization model:", list(model_options.keys()))
selected_model_name = model_options[selected_model_label]

# === Summary length settings ===
max_len = st.slider("📏 Max Summary Length", 30, 300, 120)
min_len = st.slider("📐 Min Summary Length", 5, max_len - 5, 30)

# === Setup NLTK ===
nltk.download("punkt")

# === Load summarizer with feedback ===
@st.cache_resource
def load_summarizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

with st.spinner("⏳ Loading model... (If this is your first time, downloading may take a few minutes)"):
    summarizer = load_summarizer(selected_model_name)
st.success("✅ Model is ready.")

# === Read PDF ===
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# === Chunking text ===
def chunk_text(text, tokenizer, max_chunk_tokens=512):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + " " + sentence)) < max_chunk_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# === Summarization function ===
def summarize_text(text):
    tokenizer = summarizer.tokenizer
    chunks = chunk_text(text, tokenizer)
    summaries = []

    for i, chunk in enumerate(chunks):
        try:
            with st.spinner(f"✂️ Summarizing chunk {i+1}/{len(chunks)}..."):
                summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
                summaries.append(summary)
        except Exception as e:
            st.warning(f"⚠️ Skipped chunk {i+1} due to error: {e}")
    return "\n\n".join(summaries)

# === File uploader and UI ===
pdf_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

if pdf_file:
    raw_text = read_pdf(pdf_file)
    st.text_area("📖 Preview of the document", raw_text[:2000] + "...", height=250)

    if st.button("🚀 Summarize PDF"):
        summary = summarize_text(raw_text)
        st.subheader("✅ Summary")
        st.text_area("Summary Output", summary, height=400)
