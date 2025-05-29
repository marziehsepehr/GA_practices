import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
import config_loader

# Load config
config = config_loader.load_config("config.py")
CACHE_DIR = getattr(config, "CACHE_DIR")
DEFAULT_MODEL = getattr(config, "MODEL_NAME",)
DEFAULT_MAX_LEN = getattr(config, "MAX_SUMMARY_LENGTH", 120)
DEFAULT_MIN_LEN = getattr(config, "MIN_SUMMARY_LENGTH", 30)


# === Setup NLTK ===
NLTK_DATA_DIR = os.path.join(CACHE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.download("punkt", download_dir=NLTK_DATA_DIR)
nltk.download('punkt_tab',download_dir=NLTK_DATA_DIR)
nltk.data.path.append(NLTK_DATA_DIR)

# Streamlit settings
st.set_page_config(page_title="LLM Summarizer", layout="wide")
st.title("📚 PDF Summarizer with HuggingFace LLM")

# Model options
model_options = {
    "BART Large CNN": "facebook/bart-large-cnn",
    "T5 small": "t5-small",
    "Falcon (Tiny)": "Falconsai/text_summarization"
}

selected_model_name = st.selectbox("🤖 Choose model", list(model_options.values()), index=list(model_options.values()).index(DEFAULT_MODEL))
max_len = st.slider("✂️ Max Summary Length", 30, 300, DEFAULT_MAX_LEN)
min_len = st.slider("📏 Min Summary Length", 5, max_len - 5, DEFAULT_MIN_LEN)

# Load model with cache and progress
@st.cache_resource
def load_summarizer_with_progress(model_name, cache_dir):
    progress_bar = st.progress(0, text="🚀 Downloading model files...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    progress_bar.progress(40, "🔤 Tokenizer loaded")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    progress_bar.progress(80, "🧠 Model loaded")

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    progress_bar.progress(100, "✅ Summarizer ready")
    return summarizer

# File upload
pdf_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

# Summarizer loading
summarizer = load_summarizer_with_progress(selected_model_name, CACHE_DIR)

# PDF processing
def read_pdf_text(file):
    reader = PdfReader(file)
    total_pages = len(reader.pages)
    progress_bar = st.progress(0, text="📄 Reading PDF pages...")
    text = ""

    for i, page in enumerate(reader.pages):
        text += page.extract_text() or ""
        progress_bar.progress((i + 1) / total_pages, text=f"📄 Reading PDF pages... ({i+1}/{total_pages})")

    progress_bar.empty()
    return text

def summarize_text(text, summarizer, max_len=120, min_len=30):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    max_chunk_len = 512

    # Split into chunks
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_chunk_len:
            chunk += " " + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())

    progress_bar = st.progress(0, text="📝 Summarizing text chunks...")
    summaries = []

    for i, chunk in enumerate(chunks):
        try:
            input_len = len(summarizer.tokenizer.encode(chunk, truncation=True))
            max_l = min(int(input_len * 0.5), max_len)
            min_l = min_len
            if max_l <= min_l:
                min_l = max(5, int(max_l * 0.5))

            summary = summarizer(chunk, max_length=max_l, min_length=min_l, do_sample=False)
            summaries.append(summary[0]["summary_text"])
        except Exception as e:
            st.warning(f"⚠️ Skipping a chunk due to error: {e}")
            continue

        progress_bar.progress((i + 1) / len(chunks), text=f"📝 Summarizing text chunks... ({i+1}/{len(chunks)})")

    progress_bar.empty()
    return " ".join(summaries)


# Run
if pdf_file:
    with st.spinner("📖 Reading PDF..."):
        raw_text = read_pdf_text(pdf_file)

    with st.spinner("📝 Summarizing..."):
        final_summary = summarize_text(raw_text, summarizer, max_len, min_len)

    st.subheader("📄 Summary")
    st.write(final_summary)
