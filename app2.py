
import html
import re
import time
from datetime import datetime

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from keybert import KeyBERT
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from sklearn.feature_extraction.text import TfidfVectorizer

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
SUMMARY_LENGTH_OPTIONS = [100, 150, 200, 300]


CHUNK_WORDS = 280          
BATCH_SIZE = 4            
TOKENS_PER_WORD = 1.35     
NUM_BEAMS = 1              
NOT_DETECTED = "Section not clearly detected in this document."



@st.cache_resource(show_spinner=False)
def load_summarizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    model.to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource(show_spinner=False)
def load_keyword_model():
    return KeyBERT(model="all-MiniLM-L6-v2")




def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file using PyMuPDF, keeping line breaks."""
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
        text += "\n"
    doc.close()
    return text


def strip_artifacts(text: str) -> str:
    """Remove PDF-font bullet glyphs (Private Use Area chars) and control chars
    that render as boxes in the UI."""
    text = re.sub(r"[\uf000-\uf8ff]", "- ", text) 
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  
    text = text.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
    return text


def make_structured_text(raw_text: str) -> str:
    """Cleaned text that KEEPS line breaks, so headings stay on their own line.
    Used for section detection."""
    text = strip_artifacts(raw_text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln != ""]
    return "\n".join(lines)


def make_flat_text(structured_text: str) -> str:
    """Fully whitespace-collapsed text, used for summarization/keyword extraction."""
    return re.sub(r"\s+", " ", structured_text).strip()



HEADING_CATEGORIES = {
    "methodology": [
        "methodology", "methods", "method", "approach", "proposed method",
        "proposed approach", "proposed system", "materials and methods",
        "system design", "implementation", "model architecture", "architecture",
        "proposed model", "proposed framework", "framework", "model description",
        "system architecture", "network architecture", "experimental design",
        "research design",
    ],
    "future_work": [
        "future work", "future scope", "future directions",
        "limitations and future work",
    ],
    "conclusion": [
        "conclusion", "conclusions", "conclusion and future work",
        "summary and conclusion", "concluding remarks", "closing remarks",
        "discussion and conclusion", "discussion and conclusions",
        "conclusion and discussion", "conclusions and future work",
    ],
    "stop": [
        "references", "acknowledgment", "acknowledgement", "acknowledgments",
        "acknowledgements", "appendix",
        "bibliography", "author contributions", "declaration",
    ],
}


_NUMBER_PREFIX = r"^\s*(?:[0-9]{1,2}(?:\.[0-9]{1,2})*\.?|[IVXLC]+\.)?\s*"


def _heading_regex_for(phrases):
    alternation = "|".join(re.escape(p) for p in phrases)
    return re.compile(_NUMBER_PREFIX + r"(" + alternation + r")\s*$", re.IGNORECASE)


_CATEGORY_PATTERNS = {cat: _heading_regex_for(phrases) for cat, phrases in HEADING_CATEGORIES.items()}


def find_headings(structured_text: str):
    """Scan line-by-line and return a list of (char_offset, category) for every
    heading-like line found, in document order."""
    headings = []
    offset = 0
    for line in structured_text.split("\n"):
        stripped = line.strip()
        if 0 < len(stripped) <= 60:
            for category, pattern in _CATEGORY_PATTERNS.items():
                if pattern.match(stripped):
                    headings.append((offset, category))
                    break
        offset += len(line) + 1 
    return headings


def trim_to_sentence(text: str, max_chars: int) -> str:
    """Cut text to the last full sentence within max_chars, instead of a hard
    mid-word/mid-sentence chop."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    matches = list(re.finditer(r"[.!?](\s|$)", truncated))
    if matches:
        return truncated[:matches[-1].end()].strip()
    return truncated.strip() + "..."


_TITLE_SMALL_WORDS = {"and", "of", "for", "the", "in", "to", "a", "an", "on", "with", "via", "using", "or", "vs"}
_TOP_NUM_INLINE = re.compile(r"^\s*(?:\d{1,2}|[IVXL]{1,5})\.?\s+(\S.*)$")
_TOP_NUM_ALONE = re.compile(r"^\s*(?:\d{1,2}|[IVXL]{1,5})\.?\s*$")
_ANY_NUM_INLINE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+(\S.*)$")
_ANY_NUM_ALONE = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s*$")

_METHOD_HINTS = (
    "method", "approach", "model", "architecture", "framework", "algorithm",
    "technique", "design", "system", "network", "implementation", "formulation",
    "procedure",
)
_METHOD_EXCLUDE = (
    "related", "background", "result", "experiment", "evaluation", "introduction",
    "discussion", "conclusion", "analysis", "dataset", "prior", "literature",
    "preliminar",
)
_FUTURE_CUES = (
    "future work", "future research", "future direction", "future stud", "in the future",
    "in future", "we plan to", "we intend to", "we hope to", "we would like to",
    "are excited about the future", "left for future", "leave for future",
    "remains to be", "further research", "further work", "further investigation",
    "could be extended", "can be extended",
)
_FUTURE_STRONG_CUES = (
    "future work", "future research", "future direction", "left for future", "leave for future",
)
_METHOD_CUES = (
    "we propose", "we introduce", "we present", "we develop", "we design",
    "our model", "our approach", "our method", "the proposed",
)


def _is_title_like(text: str) -> bool:
    """True for short Title Case lines such as 'Model Architecture' or 'Why Self-Attention'."""
    text = text.strip()
    words = text.split()
    if not 1 <= len(words) <= 7 or len(text) > 60:
        return False
    if text[-1] in ".,;:" or re.search(r"\d", text):
        return False
    for word in words:
        if word.lower() in _TITLE_SMALL_WORDS:
            continue
        if not (word[0].isalpha() and word[0].isupper()):
            return False
    return True


def find_top_level_headings(structured_text: str):
    """Numbered top-level section headings, e.g. '3 Model Architecture', or
    '3' on one line followed by 'Model Architecture' on the next (common in
    PDFs). Sub-sections such as '3.1' are ignored. Returns (offset, title)."""
    results = []
    offset = 0
    previous_was_number = False
    for line in structured_text.split("\n"):
        stripped = line.strip()
        title = None
        match = _TOP_NUM_INLINE.match(stripped)
        if match and _is_title_like(match.group(1)):
            title = match.group(1)
        elif previous_was_number and _is_title_like(stripped):
            title = stripped
        if title:
            results.append((offset, title))
        previous_was_number = bool(_TOP_NUM_ALONE.match(stripped))
        offset += len(line) + 1
    return results


def _find_method_heading(top_level_headings):
    """Fallback when no heading is literally called Methodology: take the first
    numbered section whose title sounds like a method description."""
    for offset, title in top_level_headings:
        lowered = title.lower()
        if any(word in lowered for word in _METHOD_EXCLUDE):
            continue
        if any(word in lowered for word in _METHOD_HINTS):
            return offset
    return None


def _strip_heading_lines(lines):
    """Remove section numbers and sub-section titles ('3.1', 'Encoder and
    Decoder Stacks') that were captured inside a section body."""
    kept = []
    previous_was_number = False
    for line in lines:
        stripped = line.strip()
        if _ANY_NUM_ALONE.match(stripped):
            previous_was_number = True
            continue
        match = _ANY_NUM_INLINE.match(stripped)
        if (match and _is_title_like(match.group(1))) or (previous_was_number and _is_title_like(stripped)):
            previous_was_number = False
            continue
        previous_was_number = False
        kept.append(stripped)
    return kept


def _cue_sentences(sentences, cues, max_sentences: int = 3, max_chars: int = 900) -> str:
    picked, total = [], 0
    for sentence in sentences:
        lowered = sentence.lower()
        if any(cue in lowered for cue in cues):
            if picked and total + len(sentence) > max_chars:
                break
            picked.append(sentence)
            total += len(sentence) + 1
            if len(picked) >= max_sentences:
                break
    return " ".join(picked)


def _fallback_section(structured_text: str, category: str) -> str:
    """Last resort when the paper has no usable heading. Future work is
    usually a few sentences at the end of the paper ('we plan to ...') and the
    method is usually announced early ('we propose ...')."""
    body = make_flat_text(strip_back_matter(structured_text))
    sentences = split_sentences(body)
    count = len(sentences)
    text = ""
    if category == "future_work":
        text = _cue_sentences(sentences[int(count * 0.7):], _FUTURE_CUES)
        if not text:
            text = _cue_sentences(sentences, _FUTURE_STRONG_CUES)
    elif category == "methodology":
        text = _cue_sentences(sentences[:int(count * 0.6)], _METHOD_CUES)
    return text if len(text) >= 30 else NOT_DETECTED


def find_section(structured_text: str, category: str, max_chars: int = 900) -> str:
    """Locate the section for `category`.
    1. Look for a heading with a known name (Methods, Future Work, Conclusion...).
    2. For methodology, fall back to the first numbered section that sounds like
       a method description (for example 'Model Architecture').
    3. Otherwise assemble the section from sentences that carry the usual cues.
    The text runs to the next heading and is trimmed to a clean sentence end."""
    headings = find_headings(structured_text)
    top_level = find_top_level_headings(structured_text)
    target_positions = [pos for pos, cat in headings if cat == category]

    if not target_positions and category == "methodology":
        method_pos = _find_method_heading(top_level)
        if method_pos is not None:
            target_positions = [method_pos]

    if not target_positions:
        return _fallback_section(structured_text, category)

    start = target_positions[0]
    boundaries = sorted(
        [pos for pos, _ in headings if pos > start] + [pos for pos, _ in top_level if pos > start]
    )
    end = boundaries[0] if boundaries else len(structured_text)
    end = min(end, start + max_chars + 400)  

    
    newline_after_heading = structured_text.find("\n", start)
    content_start = newline_after_heading + 1 if newline_after_heading != -1 else start

    lines = structured_text[content_start:end].split("\n")
    snippet = trim_to_sentence(" ".join(_strip_heading_lines(lines)), max_chars)
    if len(snippet) < 30:
        return _fallback_section(structured_text, category)
    return snippet


def strip_back_matter(structured_text: str) -> str:
    """Drop references, acknowledgements and appendices. They add a lot of
    words, none of which belong in a summary."""
    min_pos = int(len(structured_text) * 0.3)  # ignore stray matches near the top
    for pos, category in find_headings(structured_text):
        if category == "stop" and pos >= min_pos:
            return structured_text[:pos]
    return structured_text



_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def split_sentences(flat_text: str):
    """Split into sentences and discard ones that are clearly not prose
    (table rows, equation fragments, page furniture)."""
    sentences = []
    for raw in _SENTENCE_SPLIT.split(flat_text):
        sentence = raw.strip()
        if len(sentence.split()) > 70:
            
            idx = sentence.lower().rfind("abstract")
            if idx != -1:
                sentence = sentence[idx + len("abstract"):].lstrip(" :.-").strip()
        n_words = len(sentence.split())
        if n_words < 8 or n_words > 70:
            continue
        if sum(ch.isalpha() for ch in sentence) / len(sentence) < 0.6:
            continue
        sentences.append(sentence)
    return sentences


def select_key_sentences(flat_text: str, word_budget: int):
    """Extractive pre-pass. Score each sentence by how close it is to the
    centroid of the paper's TF-IDF space, favour the opening (abstract and
    introduction) and closing (conclusion) of the paper, and keep the best
    sentences up to `word_budget` words, in their original order."""
    sentences = split_sentences(flat_text)
    if not sentences:
        return [" ".join(flat_text.split()[:word_budget])]

    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= word_budget:
        return sentences

    n = len(sentences)
    try:
        tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sentences)
        centroid = np.asarray(tfidf.mean(axis=0))
        scores = np.asarray(tfidf.dot(centroid.T)).ravel()
    except ValueError:
        scores = np.ones(n)

    positions = np.arange(n)
    weights = 1.0 + 0.5 * np.clip(1.0 - positions / max(1.0, 0.12 * n), 0.0, 1.0)
    weights[positions >= int(0.92 * n)] *= 1.15

    chosen, used = [], 0
    for i in np.argsort(-(scores * weights)):
        words = len(sentences[i].split())
        if used + words > word_budget:
            continue
        chosen.append(int(i))
        used += words
        if used >= word_budget * 0.97:
            break
    chosen.sort()
    return [sentences[i] for i in chosen]


def group_into_chunks(sentences, max_words: int = CHUNK_WORDS):
    groups, current, count = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            groups.append((current, count))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        groups.append((current, count))

   
    if len(groups) > 1 and groups[-1][1] < 80:
        tail, _ = groups.pop()
        groups[-1][0].extend(tail)
    return [" ".join(group) for group, _ in groups]


def generate_summaries(tokenizer, model, device, texts, max_new_tokens: int):
    min_new_tokens = max(10, int(max_new_tokens * 0.5))
    outputs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        with torch.inference_mode():
            ids = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_length=0,  
                min_new_tokens=min_new_tokens,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=NUM_BEAMS > 1,
            )
        outputs.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return [o.strip() for o in outputs]


def summarize_text(tokenizer, model, device, flat_text: str, target_words: int):
    """Returns (summary, key_sentences). `key_sentences` is the condensed text
    the summary was built from, which is also a good input for keywords."""
    word_budget = 500 + 2 * target_words
    key_sentences = select_key_sentences(flat_text, word_budget)
    chunks = group_into_chunks(key_sentences)

    target_tokens = int(target_words * TOKENS_PER_WORD)
    per_chunk = int(min(150, max(35, target_tokens / len(chunks))))
    if len(chunks) == 1:
        per_chunk = target_tokens

    partials = generate_summaries(tokenizer, model, device, chunks, per_chunk)
    combined = " ".join(p for p in partials if p)

    
    if len(chunks) > 1 and len(combined.split()) > target_words * 1.25:
        combined = generate_summaries(tokenizer, model, device, [combined], target_tokens)[0]
    return combined, key_sentences




def extract_keywords(kw_model, text: str, top_n: int = 10):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
    )
    return [kw for kw, score in keywords]



_PUNCTUATION_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2026": "...",
    "\u2022": "-", "\u00a0": " ", "\u2009": " ", "\u202f": " ",
})


def _prepare_for_pdf(text: str) -> str:
    """fpdf's built-in fonts only support latin-1. Swap common typographic
    characters for plain equivalents, replace anything else, and break up any
    unbroken run of 60+ characters (URLs, hashes) so line wrapping always has
    somewhere to wrap."""
    text = text.translate(_PUNCTUATION_MAP)
    text = text.encode("latin-1", "replace").decode("latin-1")
    return re.sub(r"(\S{60})(?=\S)", r"\1 ", text)


def generate_summary_pdf(filename, target_words, summary, keywords, methodology, future_work, conclusion) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    width = pdf.epw  

   
    def write(text, size, style="", line_height=6, color=(0, 0, 0)):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.multi_cell(width, line_height, _prepare_for_pdf(text or ""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    write("Research Paper Summary Report", 17, "B", 9)
    pdf.ln(1)
    write(f"Source file: {filename}", 10, "", 5.5, (90, 90, 90))
    write(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}, target length about {target_words} words",
        10, "", 5.5, (90, 90, 90),
    )
    pdf.ln(3)
    pdf.set_draw_color(190, 190, 190)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    def section(title, body):
        write(title, 12, "B", 7, (31, 92, 77))
        write(body or "Not available.", 10.5, "", 5.6)
        pdf.ln(4)

    section(f"Summary (about {target_words} words)", summary)
    section("Keywords", ", ".join(keywords))
    section("Methodology", methodology)
    section("Future work", future_work)
    section("Conclusion", conclusion)

    return bytes(pdf.output())


APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@400;600&display=swap');

:root {
  --rs-accent: #1f5c4d;
  --rs-accent-dark: #174638;
  --rs-line: rgba(128, 128, 128, 0.32);
  --rs-sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
  --rs-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
}

html, body, .stApp, button, input, textarea,
[data-testid="stMarkdownContainer"], [data-baseweb="tab"] {
  font-family: var(--rs-sans) !important;
}

div.block-container, [data-testid="stMainBlockContainer"] {
  max-width: 1120px !important;
  margin: 0 auto;
  padding-top: 2.75rem !important;
  padding-bottom: 4rem !important;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Page header */
.rs-title { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; line-height: 1.2; margin: 0; }
.rs-sub { margin: 0.4rem 0 1.5rem; opacity: 0.72; font-size: 0.98rem; max-width: 62ch; line-height: 1.5; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.75rem; }
.stTabs [data-baseweb="tab"] { padding: 0.55rem 0; height: auto; font-weight: 500; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--rs-accent); height: 2px; }
.stTabs [data-baseweb="tab-border"] { background-color: var(--rs-line); }

/* Input panel */
[data-testid="stColumn"]:has(.rs-panel-marker),
[data-testid="column"]:has(.rs-panel-marker) {
  background: rgba(128, 128, 128, 0.09);
  border-radius: 6px;
  padding: 1.25rem 1.25rem 1.5rem;
  box-sizing: border-box;
  align-self: flex-start;
}

/* Buttons: square-cornered, solid, no pills */
.stApp button { border-radius: 4px !important; }
.stApp button[kind="primary"],
.stApp button[data-testid="stBaseButton-primary"] {
  background-color: var(--rs-accent);
  border: 1px solid var(--rs-accent);
  color: #ffffff;
  padding: 0.5rem 1.4rem;
  font-weight: 500;
}
.stApp button[kind="primary"] p,
.stApp button[data-testid="stBaseButton-primary"] p { color: #ffffff; }
.stApp button[kind="primary"]:hover:not(:disabled),
.stApp button[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
  background-color: var(--rs-accent-dark);
  border-color: var(--rs-accent-dark);
  color: #ffffff;
}
.stApp button[kind="primary"]:disabled,
.stApp button[data-testid="stBaseButton-primary"]:disabled { opacity: 0.4; }
.stApp button[kind="secondary"],
.stApp button[data-testid="stBaseButton-secondary"] { border: 1px solid var(--rs-line); font-weight: 500; }

[data-testid="stFileUploaderDropzone"] { border-radius: 4px; }
[data-testid="stExpander"] { border-radius: 4px; border: 1px solid var(--rs-line); }

/* Results */
.rs-file { font-size: 1.2rem; font-weight: 600; margin: 0 0 0.9rem; word-break: break-word; }
.rs-meta {
  display: flex; flex-wrap: wrap; gap: 0.6rem 2.25rem;
  padding: 0.75rem 0; margin-bottom: 1rem;
  border-top: 1px solid var(--rs-line); border-bottom: 1px solid var(--rs-line);
}
.rs-meta-item { display: flex; flex-direction: column; }
.rs-meta-key { font-size: 0.78rem; opacity: 0.65; }
.rs-meta-val { font-size: 0.98rem; font-weight: 500; }

.rs-h { font-size: 1rem; font-weight: 600; margin: 1.8rem 0 0.5rem; }
.rs-h:first-child { margin-top: 0.4rem; }
.rs-note { font-weight: 400; font-size: 0.82rem; opacity: 0.6; margin-left: 0.7rem; }

.rs-prose { font-family: var(--rs-serif); font-size: 1.05rem; line-height: 1.72; max-width: 68ch; margin: 0; }
.rs-extract { border-left: 2px solid var(--rs-accent); padding-left: 0.9rem; }
.rs-muted { opacity: 0.55; font-style: italic; }

.rs-tags { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.rs-tag { border: 1px solid var(--rs-line); border-radius: 2px; padding: 0.15rem 0.55rem; font-size: 0.86rem; }

.rs-empty { padding: 0.4rem 0 1rem; opacity: 0.72; max-width: 54ch; line-height: 1.6; }
.rs-preview { font-size: 0.9rem; line-height: 1.6; opacity: 0.85; word-break: break-word; }

/* Downloads list */
.rs-row-name { font-weight: 500; word-break: break-word; }
.rs-row-meta { font-size: 0.85rem; opacity: 0.65; margin-top: 0.15rem; }
.rs-rule { border-top: 1px solid var(--rs-line); margin: 0.5rem 0 0.9rem; }
</style>
"""


def _h(text) -> str:
    """Escape text for safe inclusion in HTML passed to st.markdown. Whitespace is
    collapsed and '$' is escaped so Streamlit never treats it as a math delimiter."""
    return html.escape(" ".join(str(text).split())).replace("$", "&#36;")


def _fmt_seconds(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f} seconds"
    if seconds < 90:
        return f"{seconds:.0f} seconds"
    return f"{seconds / 60:.1f} minutes"




def process_paper(uploaded_file, target_words):
    started = time.perf_counter()

    with st.status("Processing paper", expanded=True) as status:
        status.write("Extracting text from the PDF")
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
        except Exception:
            status.update(label="Could not read the file", state="error")
            st.error("This file could not be opened as a PDF. Check that it is not corrupted or password protected.")
            return None

        structured_text = make_structured_text(raw_text)
        flat_text = make_flat_text(structured_text)
        body_text = make_flat_text(strip_back_matter(structured_text))

        if len(flat_text.split()) < 50:
            status.update(label="No readable text found", state="error")
            st.error("Not enough text could be extracted. The PDF may be scanned or image-based and would need OCR first.")
            return None

        status.write("Summarizing")
        tokenizer, model, device = load_summarizer()
        summary, key_sentences = summarize_text(tokenizer, model, device, body_text, target_words)

        status.write("Extracting keywords")
        keywords = extract_keywords(load_keyword_model(), " ".join(key_sentences))

        status.write("Locating methodology, future work and conclusion")
        methodology = find_section(structured_text, "methodology")
        future_work = find_section(structured_text, "future_work")
        conclusion = find_section(structured_text, "conclusion")

        status.write("Building the PDF report")
        pdf_bytes = generate_summary_pdf(
            uploaded_file.name, target_words, summary, keywords, methodology, future_work, conclusion
        )

        elapsed = time.perf_counter() - started
        status.update(label=f"Finished in {_fmt_seconds(elapsed)}", state="complete", expanded=False)

    return {
        "filename": uploaded_file.name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target_words": target_words,
        "word_count": len(flat_text.split()),
        "elapsed": elapsed,
        "summary": summary,
        "keywords": keywords,
        "methodology": methodology,
        "future_work": future_work,
        "conclusion": conclusion,
        "raw_text_preview": flat_text[:3000],
        "pdf_bytes": pdf_bytes,
    }


def _extracted_block(title, text):
    if text == NOT_DETECTED:
        return f'<div class="rs-h">{title}</div><div class="rs-prose rs-muted">{_h(text)}</div>'
    return (
        f'<div class="rs-h">{title}<span class="rs-note">extracted from the paper</span></div>'
        f'<div class="rs-prose rs-extract">{_h(text)}</div>'
    )


def render_result(result):
    meta = [
        ("Words extracted", f"{result['word_count']:,}"),
        ("Target length", f"about {result['target_words']} words"),
        ("Processing time", _fmt_seconds(result["elapsed"])),
        ("Generated", result["timestamp"]),
    ]
    meta_html = "".join(
        f'<div class="rs-meta-item"><span class="rs-meta-key">{key}</span>'
        f'<span class="rs-meta-val">{_h(value)}</span></div>'
        for key, value in meta
    )
    st.markdown(
        f'<div class="rs-file">{_h(result["filename"])}</div><div class="rs-meta">{meta_html}</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download PDF report",
        data=result["pdf_bytes"],
        file_name=f"{result['filename'].rsplit('.', 1)[0]}_summary.pdf",
        mime="application/pdf",
        key="download_current",
    )

    tags = "".join(f'<span class="rs-tag">{_h(kw)}</span>' for kw in result["keywords"])
    body = (
        f'<div class="rs-h">Summary<span class="rs-note">generated, {len(result["summary"].split())} words</span></div>'
        f'<div class="rs-prose">{_h(result["summary"])}</div>'
        f'<div class="rs-h">Keywords</div><div class="rs-tags">{tags}</div>'
        + _extracted_block("Methodology", result["methodology"])
        + _extracted_block("Future work", result["future_work"])
        + _extracted_block("Conclusion", result["conclusion"])
    )
    st.markdown(body, unsafe_allow_html=True)

    st.write("")
    with st.expander("Extracted text preview"):
        st.markdown(f'<div class="rs-preview">{_h(result["raw_text_preview"])}...</div>', unsafe_allow_html=True)


def render_empty_state():
    st.markdown(
        '<div class="rs-empty">Results appear here once a paper has been processed. '
        "You get a summary at the length you choose, the main keywords, and the "
        "methodology, future work and conclusion sections taken from the paper.</div>",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Research Paper Summarizer", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="rs-title">Research paper summarizer</div>'
        '<div class="rs-sub">Upload a research paper as a PDF to get a summary, keywords, '
        "and its methodology, future work and conclusion sections.</div>",
        unsafe_allow_html=True,
    )

    if "history" not in st.session_state:
        st.session_state.history = []  
    if "current_result" not in st.session_state:
        st.session_state.current_result = None

    
    with st.spinner("Loading language models. This is only slow the first time."):
        load_summarizer()
        load_keyword_model()

    tab_summarize, tab_downloads = st.tabs(["Summarize", "Downloads"])

    with tab_summarize:
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown('<div class="rs-panel-marker" style="display:none"></div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Research paper (PDF)", type=["pdf"])
            target_words = st.select_slider(
                "Summary length in words",
                options=SUMMARY_LENGTH_OPTIONS,
                value=150,
                help="Roughly how long the generated summary should be.",
            )
            generate_clicked = st.button("Generate summary", type="primary", disabled=uploaded_file is None)

            if generate_clicked and uploaded_file is not None:
                result = process_paper(uploaded_file, target_words)
                if result is not None:
                    st.session_state.current_result = result
                    st.session_state.history.insert(0, result)

        with right:
            if st.session_state.current_result is not None:
                render_result(st.session_state.current_result)
            else:
                render_empty_state()

    with tab_downloads:
        if not st.session_state.history:
            st.markdown(
                '<div class="rs-empty">No reports yet. Papers you summarize on the Summarize tab are '
                "listed here, and they stay available after you upload a different file.</div>",
                unsafe_allow_html=True,
            )
        else:
            for i, item in enumerate(st.session_state.history):
                info, action = st.columns([4, 1], gap="medium")
                with info:
                    st.markdown(
                        f'<div class="rs-row-name">{_h(item["filename"])}</div>'
                        f'<div class="rs-row-meta">Generated {_h(item["timestamp"])}, '
                        f'about {item["target_words"]} words, '
                        f'processed in {_fmt_seconds(item["elapsed"])}</div>',
                        unsafe_allow_html=True,
                    )
                with action:
                    st.download_button(
                        "Download PDF",
                        data=item["pdf_bytes"],
                        file_name=f"{item['filename'].rsplit('.', 1)[0]}_summary.pdf",
                        mime="application/pdf",
                        key=f"download_{i}",
                    )
                st.markdown('<div class="rs-rule"></div>', unsafe_allow_html=True)

            if st.button("Clear history"):
                st.session_state.history = []
                st.session_state.current_result = None
                st.rerun()


if __name__ == "__main__":
    main()
