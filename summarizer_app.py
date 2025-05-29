import os
import nltk
import PyPDF2
import config_loader
import streamlit as st
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
st.set_page_config(page_title="PDF Summarizer with Local LLM", layout="wide")

# === Load Config ===
config = config_loader.load_config("config.py")
CACHE_DIR = getattr(config, "CACHE_DIR", None)
MODEL_NAME = getattr(config, "MODEL_NAME", None)
TRANSFORMERS_OFFLINE = getattr(config, "TRANSFORMERS_OFFLINE", "1")

os.environ["TRANSFORMERS_OFFLINE"] = TRANSFORMERS_OFFLINE
nltk.data.path.append(os.path.join(CACHE_DIR, "nltk_data"))
nltk.download("punkt", download_dir=os.path.join(CACHE_DIR, "nltk_data"), quiet=True)

# === Load Model ===
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(CACHE_DIR, MODEL_NAME))
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(CACHE_DIR, MODEL_NAME))
    return pipeline("summarization", model=model, tokenizer=tokenizer, device=0)

summarizer = load_summarizer()

# === Helper Functions ===
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

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

def summarize_text(text):
    chunks = chunk_text(text, summarizer.tokenizer)
    summaries = []

    for i, chunk in enumerate(chunks):
        input_len = len(summarizer.tokenizer.encode(chunk))
        max_len = min(int(input_len * 0.6), 120)
        min_len = max(10, int(max_len * 0.5))

        try:
            with st.spinner(f"Summarizing chunk {i+1} of {len(chunks)}..."):
                summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
                summaries.append(summary)
        except Exception as e:
            st.warning(f"Skipped chunk {i+1} due to error: {e}")
    return "\n\n".join(summaries)

# === UI ===

st.title("📄 Summarize PDF Using Local Language Model")

pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file:
    raw_text = read_pdf(pdf_file)
    st.text_area("📚 PDF Content (Preview)", raw_text[:3000] + "...", height=300, disabled=True)

    if st.button("🔍 Summarize"):
        summary = summarize_text(raw_text)
        st.subheader("✂️ Generated Summary:")
        st.text_area("Final Summary", summary, height=400)
