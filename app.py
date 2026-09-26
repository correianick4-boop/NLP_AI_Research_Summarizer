"""
Automatic Research Paper Summarizer Using NLP
-----------------------------------------------
Demo implementation matching the project proposal:
- Accepts a research paper PDF
- Extracts text (PyMuPDF)
- Generates a ~100 word summary (Hugging Face Transformers)
- Extracts keywords (KeyBERT)
- Pulls out likely "Methodology" / "Future Work" sections via keyword search
- Displays everything in a Streamlit web interface

Run with:
    pip install streamlit pymupdf transformers torch keybert sentence-transformers
    streamlit run app.py
"""

import re
import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from keybert import KeyBERT
from transformers import AutoTokenizer


# ----------------------------
# Cached model loaders
# ----------------------------

@st.cache_resource
def load_summarizer():
    # facebook/bart-large-cnn works well for long-form summarization
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


@st.cache_resource
def load_keyword_model():
    return KeyBERT(model="all-MiniLM-L6-v2")


# ----------------------------
# PDF text extraction
# ----------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file using PyMuPDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def clean_text(text: str) -> str:
    """Basic preprocessing: collapse whitespace, drop reference-heavy tail."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


# ----------------------------
# Chunking for long documents
# ----------------------------

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def truncate_to_tokens(tokenizer, text, max_tokens=1000):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def summarize_long_text(summarizer, tokenizer, text, target_words=100):
    chunks = list(chunk_text(text))
    chunk_summaries = []
    for chunk in chunks:
        safe_chunk = truncate_to_tokens(tokenizer, chunk)
        wc = len(safe_chunk.split())
        if wc < 20:
            continue
        max_len = min(150, max(30, wc // 2))
        result = summarizer(safe_chunk, max_length=max_len, min_length=min(20, max_len - 5), do_sample=False)
        chunk_summaries.append(result[0]["summary_text"])

    combined = " ".join(chunk_summaries)
    combined = truncate_to_tokens(tokenizer, combined)

    if len(chunk_summaries) > 1:
        result = summarizer(combined, max_length=target_words + 40, min_length=max(20, target_words - 30), do_sample=False)
        return result[0]["summary_text"]
    return combined


# ----------------------------
# Keyword extraction
# ----------------------------

def extract_keywords(kw_model, text: str, top_n: int = 10):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
    )
    return [kw for kw, score in keywords]


# ----------------------------
# Section extraction (methodology / future work)
# ----------------------------

SECTION_PATTERNS = {
    "Methodology": [
        r"(methodology|methods|approach|proposed method)(.{200,1500}?)(?=\n[A-Z][a-z]+\n|\n\d\.|$)",
    ],
    "Future Work": [
        r"(future work|future scope|conclusion and future work)(.{100,1000}?)(?=\n[A-Z][a-z]+\n|\n\d\.|$)",
    ],
}


def find_section(text: str, section_key: str) -> str:
    lowered = text.lower()
    for pattern in SECTION_PATTERNS[section_key]:
        match = re.search(pattern, lowered, re.DOTALL)
        if match:
            start = match.start()
            # Grab corresponding text from the original (non-lowered) string
            snippet = text[start:start + 600]
            return snippet.strip()
    return "Section not clearly detected — showing keyword-based summary instead."


# ----------------------------
# Streamlit UI
# ----------------------------

def main():
    st.set_page_config(page_title="Research Paper Summarizer", layout="wide")
    st.title("📄 Automatic Research Paper Summarizer")
    st.caption("Upload a research paper (PDF) to get a summary, keywords, methodology, and future work.")

    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            text = clean_text(raw_text)

        if len(text.split()) < 50:
            st.error("Couldn't extract enough text. The PDF may be scanned/image-based (would need OCR).")
            return

        st.success(f"Extracted {len(text.split())} words from the PDF.")

        with st.expander("View extracted raw text"):
            st.write(text[:3000] + ("..." if len(text) > 3000 else ""))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Summary (~100 words)")
            with st.spinner("Generating summary... (first run downloads the model, can take a minute)"):
                summarizer = load_summarizer()
                tokenizer = load_tokenizer()
                summary = summarize_long_text(summarizer, tokenizer, text, target_words=100)
            st.write(summary)

            st.subheader("🔑 Keywords")
            with st.spinner("Extracting keywords..."):
                kw_model = load_keyword_model()
                keywords = extract_keywords(kw_model, text)
            st.write(", ".join(keywords))

        with col2:
            st.subheader("🧪 Methodology (detected)")
            st.write(find_section(text, "Methodology"))

            st.subheader("🔮 Future Work (detected)")
            st.write(find_section(text, "Future Work"))

        st.download_button(
            "Download Summary as Text",
            data=f"SUMMARY:\n{summary}\n\nKEYWORDS:\n{', '.join(keywords)}",
            file_name="paper_summary.txt",
        )


if __name__ == "__main__":
    main()