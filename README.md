# StructRead — Proof of Concept

Structural offloading for reading persistence in adults with ADHD.  
Same words, new layout — the LLM labels discourse structure, the renderer externalizes it.

## Quick start (local)

```bash
# 1. Clone or copy this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Enter your **Groq API key** in the sidebar (free at [console.groq.com](https://console.groq.com)).

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
Text input
    │
    ▼
┌──────────────────┐
│  Sentence Split   │  Python (regex-based)
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Groq LLM API    │  Structural annotation only
│  (labels only)   │  No text generation
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Label Parser     │  JSON → validated label list
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  HTML Renderer    │  Original text + labels → formatted page
│  (rendering      │  Visual encoding:
│   engine)        │    • List markers = coordination
└──────────────────┘    • Indentation = subordination
    │                   • Vertical rule = deferrable
    ▼                   • Whitespace = unit boundaries
  Browser
```

## Visual encoding scheme

| Visual channel       | Structural role          |
|----------------------|--------------------------|
| List markers (1. 2.) | Coordinate/parallel ideas |
| Indentation          | Hierarchical subordination |
| Left vertical rule   | Deferrable detail         |
| Inter-block whitespace | Structural unit boundary |
| Bold text            | Topic sentence            |
| Italic               | Transition                |

## Files

- `app.py` — Complete Streamlit app (prompt + API + renderer + UI)
- `requirements.txt` — Python dependencies
- `README.md` — This file
