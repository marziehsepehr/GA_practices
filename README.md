# LLM Practice Suite

This project is a modular playground for experimenting with Large Language Models (LLMs) on various NLP tasks. It currently supports book/document summarization and is designed for easy extension to other tasks such as Named Entity Recognition (NER), Question Answering (QA), and more.

## Features

- **Summarization:** Chunking and step-by-step summarization of large documents using local HuggingFace models.
- **PDF Support:** Reads and processes PDF files.
- **Interactive Web App:** Summarize PDFs directly in your browser using the Streamlit-based `summarizer_app.py`.
- **Configurable:** All model and cache paths set via `config.py`.
- **Extensible:** Structure supports adding new LLM-based modules (NER, QA, etc.).

## Planned Modules

- [x] Summarization (Jupyter notebook & Streamlit app)
- [ ] Named Entity Recognition (NER)
- [ ] Question Answering (QA)
- [ ] Custom LLM Experiments

## Folder Structure

```
.
├── config.py             # Configuration file for model/cache settings
├── config_loader.py      # Loads config.py dynamically
├── summrizer.ipynb       # Summarization notebook
├── summarizer_app.py     # Streamlit app for PDF summarization
├── The McKinsey Way...pdf  # Example PDF book
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- Python 3.8+
- [transformers](https://huggingface.co/transformers/)
- [torch](https://pytorch.org/)
- [nltk](https://www.nltk.org/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [streamlit](https://streamlit.io/)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Configure paths:**  
   Edit `config.py` to set your model and cache directory.

2. **Download NLTK data:**  
   The notebook will automatically download required NLTK data to your cache directory.

3. **Run a module:**  
   - For the Jupyter notebook, open `summrizer.ipynb` in Jupyter or VS Code and run all cells.
   - For the Streamlit app, run `streamlit run summarizer_app.py` in your terminal and open the provided URL in your browser.



## How it works

- Loads configuration from `config.py`.
- Loads a local Transformers model for the selected task.
- Reads and processes input data (e.g., PDF for summarization).
- Applies chunking and LLM inference as needed.
- Designed for easy extension to new tasks.

## License

MIT License. See [LICENSE](LICENSE) for details.