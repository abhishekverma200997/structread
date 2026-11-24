# StructRead — Proof of Concept

Structural offloading for reading persistence in adults with ADHD.  
Same words, new layout — the LLM labels discourse structure, the renderer externalizes it.

## Quick start (local)

```bash
# 1. Clone or copy this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key (free at console.groq.com) — either:
#    a) .streamlit/secrets.toml containing:  GROQ_API_KEY = "gsk_your_key_here"
#    b) an environment variable:            GROQ_API_KEY=gsk_your_key_here

# 4. Run
streamlit run app.py
```

The key is read on the server only — it is never shown or entered in the UI.
`.streamlit/secrets.toml` is git-ignored.

> **Windows on ARM:** use the x64 build of Python — `pyarrow` (needed by Streamlit) has no ARM64 wheels.

## What it handles

- **Input:** pasted text, PDFs (with a page-range picker for long documents), or the built-in sample.
- **Scanned / image-only PDFs:** read with OCR (RapidOCR), including PDFs whose text was saved as shapes ("Print to PDF").
- **Figures, charts and tables in PDFs:** captured as images and shown in place — never sent to the LLM.
- **Pasted tables** (tab-separated or Markdown `| pipe |`): rendered as real HTML tables.
- **Headings, captions, equations:** detected and shown as-is.
- **List-like sections** (references, bibliography, contents, index, notes, glossary): shown exactly as written.
- **Two-column papers:** read left column, then right; running headers, footers and page numbers removed.
- **Flat text:** if most labels are low-confidence, the original layout is shown with paragraph spacing.
- **Long documents:** split into chunks with context from the previous chunk; a failed chunk is retried, then
  shown as plain text instead of losing the whole run; Groq rate limits are waited out or reported.

## Deploy & share a link

### Option A: Streamlit Community Cloud (recommended — free)

1. Push this folder to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo → select `app.py`
4. Add your Groq API key in **Settings → Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy — you get a shareable URL

`packages.txt` installs the system libraries OpenCV needs for OCR on Streamlit Cloud's Linux servers.

### Option B: ngrok (quick, temporary)

```bash
# Terminal 1
streamlit run app.py

# Terminal 2
ngrok http 8501
```

Share the ngrok URL with your professor.

## Architecture

```
Text / PDF
    │
    ▼
Extraction & segmentation   PyMuPDF (+ OCR); headings, figures, tables, references are kept
    │                       as-is and never sent to the LLM
    ▼
Sentence split → chunks     regex splitter; ≤ 40 sentences / ~1,200 tokens per chunk
    │
    ▼
Sentence centrality         MiniLM embeddings → similarity graph (prompt signal + sanity check)
    │
    ▼
Global pass (1 call)        outline, thesis, key terms — context for every chunk (2+ chunks only)
    │
    ▼
Local labeling (Groq)       one label per sentence id; labels only, no text generation
    │
    ▼
Validate → reconcile        ids checked, indent capped at 1, chunk-boundary / per-paragraph topic
    │                       conflicts resolved, bold budget (≤ 20% of sentences) enforced
    ▼
HTML renderer               original sentences + labels → page with collapsible sections
```

## Visual encoding scheme

Deliberately small: every treatment below is the author's own words in a different layout.

| Visual channel            | Structural role                         |
|---------------------------|-----------------------------------------|
| Bold                      | Topic sentence (at most ~20% of text)   |
| Numbered list             | Coordinate / parallel ideas             |
| One level of indentation  | Supporting detail                       |
| Muted text + left rule    | Skippable detail (always visible)       |
| Italic                    | Connecting sentence                     |
| Collapsible sections      | One section at a time, with progress    |

Not drawn on the page: the transition type and the "relation to parent" the model returns. They are
kept in the labels (see *Analysis details* in the Input tab) but add nothing to what the reader sees.
Low-confidence labels render as plain text.

## Development

```bash
# Tests (no API key or model download needed)
pip install pytest
pytest

# Dev mode: put a PDF at test_data/default.pdf — it is processed once, then loads instantly
STRUCTREAD_DEV=1 streamlit run app.py          # PowerShell: $env:STRUCTREAD_DEV=1; streamlit run app.py
```

Changing `SYSTEM_PROMPT` invalidates saved labels, so the next run of a document calls Groq again.

## Files

- `app.py` — Complete Streamlit app (prompt + API + renderer + UI)
- `requirements.txt` — Python dependencies
- `packages.txt` — System packages for Streamlit Community Cloud
- `tests/test_core.py` — Tests for splitting, label validation and the "same words" rendering guarantee
- `README.md` — This file
