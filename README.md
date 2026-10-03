# Automatic Research Paper Summarizer

A Streamlit app that reads a research paper PDF and returns a summary, keywords, and the methodology, future work and conclusion sections, with a downloadable PDF report for every paper.

## Why this project is different

- **Fast on a normal laptop.** The paper is condensed with an extractive pass first, so the neural model only reads the few hundred words that matter, not the whole document. Processing time is shown after every run.
- **Sections are found, not guessed.** Methodology, future work and conclusion are located with heading-aware parsing. When a paper has no standard heading (for example "Model Architecture" instead of "Methodology"), the app falls back to numbered-section matching, then to sentence cues such as "we plan to".
- **Adjustable summary length.** Choose 100, 150, 200 or 300 words.
- **Reports that persist.** Every summarized paper is kept in a Downloads tab for the session, even after you upload a different file.
- **Runs locally.** No API keys and no data leaves your machine.

## How it works

1. **Extract.** PyMuPDF reads the PDF and the text is cleaned of font artifacts.
2. **Trim.** References, acknowledgements and appendices are removed.
3. **Condense.** TF-IDF scoring keeps the most informative sentences, favouring the abstract, introduction and conclusion.
4. **Summarize.** DistilBART (`sshleifer/distilbart-cnn-12-6`) summarizes the kept sentences in one batched call with greedy decoding.
5. **Keywords.** KeyBERT (`all-MiniLM-L6-v2`) pulls the top 10 keywords and key phrases.
6. **Sections.** Headings are parsed to extract methodology, future work and conclusion.
7. **Report.** fpdf2 builds a PDF with all of the above.

## Speed

The original version summarized every 400-word slice of the full paper using beam search, which took 10 to 15 minutes on a CPU. The current pipeline summarizes roughly 700 to 1,100 selected words instead, in batches, with greedy decoding, and loads the models once at startup. The target is 1 to 2 minutes on a typical CPU, and faster with a GPU, which is used automatically if available. Actual time depends on your hardware, so check the processing time shown in the results.

## Tech stack

| Part | Tool |
| --- | --- |
| Interface | Streamlit |
| PDF reading | PyMuPDF |
| Summarization | Hugging Face Transformers, PyTorch, DistilBART |
| Sentence selection | scikit-learn (TF-IDF) |
| Keywords | KeyBERT, sentence-transformers |
| Report export | fpdf2 |

## Setup

```bash
pip install streamlit pymupdf transformers torch keybert sentence-transformers fpdf2 scikit-learn
streamlit run app1.py
```

The first launch downloads the models, so it needs an internet connection and takes a little longer.

## Usage

1. Open the Summarize tab and upload a PDF.
2. Pick a summary length.
3. Select Generate summary.
4. Read the results, or download the PDF report. Past reports are listed in the Downloads tab.

## Limitations

- Scanned or image-only PDFs are not supported because there is no OCR step.
- Section detection is heuristic. Papers with unusual layouts may show "Section not clearly detected".
- The summary is built from a condensed version of the paper, so minor details can be left out.
- History is kept in the browser session only and resets when the app restarts.

## Future work

- OCR support for scanned papers.
- Optional larger models for higher quality summaries.
- Saving history to disk.
- Batch upload of several papers at once.
