"""
StructRead — Structural Offloading for Reading Persistence
Proof-of-concept prototype

Takes any text → LLM labels discourse structure → renders the same words
in a layout that externalizes that structure.
"""

import streamlit as st
import json
import re
import io
import os
import time
import math
from groq import Groq

# ─────────────────────────────────────────────
# 1. LLM SYSTEM PROMPT — the structural annotator
# ─────────────────────────────────────────────

SYSTEM_PROMPT = r"""You are a discourse-structure annotator. Your ONLY job is to label the structural role of each sentence in a text. You must NEVER generate, rephrase, add, or remove any text.

## Input
You will receive numbered sentences like:
[1] First sentence.
[2] Second sentence.

## Output
Return ONLY a valid JSON object (no markdown, no backticks, no commentary) with this exact schema:

{
  "sentences": [
    {
      "id": 1,
      "role": "topic",
      "indent": 0,
      "group": null,
      "parent_id": null,
      "confidence": 0.92
    }
  ]
}

## Roles — assign exactly ONE per sentence

**topic** — The governing claim or main idea of a paragraph or section. Other sentences support, explain, or elaborate on it. Typically 1 per paragraph. Not every paragraph-opener is a topic sentence — only label it "topic" if subordinate content actually follows.

**coordinate** — One of several parallel ideas at the same structural level: items in a list, parallel reasons, contrasting alternatives, sequential steps. Coordinate sentences that belong together MUST share the same "group" letter (A, B, C…). A group must have at least 2 members. Look for cues: "First… Second… Third…", "One… Another… A third…", semicolon-separated claims, numbered items, or repeated syntactic patterns.

**subordinate** — Directly supports, explains, exemplifies, or provides evidence for another sentence. Set "parent_id" to the id of the sentence it depends on. Set "indent" to 1 (or 2 if nested under another subordinate).

**deferrable** — An aside, methodological detail, tangential elaboration, specific statistic restating an already-stated claim, historical note, or analogy that could be skipped without breaking the argument's logic. Set "parent_id" to the sentence it elaborates. These will be rendered as collapsible. Deferrable ≠ unimportant; it means the core argument survives without it.

**transition** — Connects two structural units. Typically contains discourse markers: "However," "Therefore," "In contrast," "Turning to," "As a result," "On the other hand." Also includes concluding/summarizing sentences that tie previous ideas together.

## Field definitions

- **id** (integer): The sentence number from the input. Must match exactly.
- **role** (string): One of: topic, coordinate, subordinate, deferrable, transition.
- **indent** (integer): Nesting depth. 0 = top level, 1 = one level in, 2 = two levels in. Maximum 2. Topic and transition are always 0. Coordinate items inherit the indent of their structural position. Subordinate/deferrable are at least 1.
- **group** (string or null): For coordinate sentences only — a shared letter (A, B, C…) identifying which parallel group they belong to. null for all other roles.
- **parent_id** (integer or null): For subordinate and deferrable only — the id of the sentence this one depends on. null for topic, coordinate, transition.
- **confidence** (float): 0.0 to 1.0. Your confidence in this specific label.

## Rules

1. PRESERVE EVERY SENTENCE. Your output must have exactly as many entries as input sentences, with matching ids. Do not skip, merge, or split any sentence.
2. NO TEXT GENERATION. You are labeling, not writing. If you catch yourself composing new words, stop.
3. WHEN UNSURE, DEFAULT SAFE. If confidence is below 0.6, label as "subordinate" with indent 0 and confidence 0.5. Wrong structural labels are worse than conservative ones.
4. COORDINATE GROUPS NEED ≥2 MEMBERS. If you can only find one item, it's not coordination — label it subordinate or topic instead.
5. WATCH FOR FALSE COORDINATION. A causal chain (A causes B causes C) is NOT coordination — it's a sequence of subordination. Only label as coordinate when items are genuinely parallel and interchangeable in order.
6. DISTINGUISH DEFERRABLE FROM SUBORDINATE. A subordinate sentence is needed to understand the argument. A deferrable sentence enriches but is not needed. When in doubt, choose subordinate.
7. TOPIC SENTENCES ARE RARE. ~1 per paragraph. If a paragraph has no clear topic sentence, the first sentence is usually subordinate to the previous paragraph's topic.
8. LOOK FOR LINGUISTIC CUES:
   - "First / Second / Third / Finally" → coordinate
   - "For example / For instance / Such as" → subordinate
   - "In other words / That is" → deferrable (restating)
   - "However / Therefore / Thus / As a result" → transition or topic
   - "Notably / Importantly / Critically" → often marks a topic or key subordinate
   - "Specifically / In particular" → subordinate
   - Parenthetical asides, em-dashes with elaboration → deferrable
"""

# ─────────────────────────────────────────────
# 2. SAMPLE TEXT for demo
# ─────────────────────────────────────────────

SAMPLE_TEXT = """Sleep deprivation affects cognitive performance through multiple interconnected pathways. The most critical pathway involves the hippocampus, which serves as the brain's primary memory consolidation center. During deep sleep, the hippocampus replays neural patterns from the day's experiences, transferring information from short-term to long-term storage. When sleep is curtailed, this replay process is truncated, resulting in fragmented memory traces that are difficult to retrieve. A second pathway operates through the prefrontal cortex, which governs executive functions including attention regulation, decision-making, and impulse control. Sleep loss reduces prefrontal activation by approximately 14%, as measured by functional MRI studies conducted between 2018 and 2023. This reduction manifests behaviorally as increased distractibility, poor task-switching, and elevated error rates on sustained attention tasks. Notably, the landmark Williamson and Feyer study demonstrated that 17 hours of sustained wakefulness produces cognitive impairment equivalent to a blood alcohol concentration of 0.05%. A third pathway involves the dysregulation of cortisol, the primary stress hormone. Under normal conditions, cortisol follows a circadian rhythm, peaking shortly after waking and declining throughout the day. Sleep deprivation disrupts this rhythm, maintaining elevated cortisol levels that, over time, damage dendritic connections in the hippocampus. These three pathways do not operate independently. Prefrontal dysfunction reduces the brain's ability to suppress irrelevant information during encoding, while elevated cortisol impairs the hippocampal consolidation that would normally compensate for noisy encoding. The result is a compounding deficit: each pathway's failure amplifies the others."""

# ─────────────────────────────────────────────
# 3. HTML + CSS TEMPLATE for rendered output
# ─────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 17px;
    line-height: 1.8;
    color: #2C2C2A;
    background: #FFFFFF;
    padding: 32px 24px;
    max-width: 720px;
    margin: 0 auto;
  }}

  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 32px;
    padding: 14px 18px;
    background: #F6F5F0;
    border-radius: 8px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 12px;
    color: #5F5E5A;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-swatch {{
    width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0;
  }}

  .s-topic {{
    margin-top: 28px;
    margin-bottom: 10px;
    font-weight: 600;
    color: #1a1a18;
  }}
  .s-topic:first-child {{ margin-top: 0; }}

  .s-transition {{
    margin: 24px 0 10px 0;
    color: #3a3a38;
    font-style: italic;
  }}

  .coord-group {{
    margin: 12px 0 12px 4px;
    padding-left: 0;
  }}
  .coord-group ol {{
    margin: 0;
    padding-left: 28px;
    list-style-type: decimal;
  }}
  .coord-group ol li {{
    margin-bottom: 10px;
    padding-left: 4px;
  }}
  .coord-group ol li::marker {{
    color: #378ADD;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}

  .s-subordinate {{
    margin-left: 32px;
    color: #3d3d3b;
    font-size: 16px;
    margin-bottom: 8px;
  }}
  .s-subordinate.indent-2 {{
    margin-left: 56px;
    font-size: 15px;
    color: #555553;
  }}

  .deferrable-wrapper {{
    margin: 6px 0 6px 20px;
  }}
  .deferrable-toggle {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 13px;
    color: #888780;
    cursor: pointer;
    user-select: none;
    padding: 4px 10px;
    border-radius: 4px;
    background: #F6F5F0;
    border: none;
    transition: background 0.15s;
  }}
  .deferrable-toggle:hover {{
    background: #EEEDEA;
    color: #5F5E5A;
  }}
  .deferrable-toggle .arrow {{
    display: inline-block;
    transition: transform 0.2s;
    font-size: 10px;
  }}
  .deferrable-toggle.open .arrow {{
    transform: rotate(90deg);
  }}
  .deferrable-body {{
    display: none;
    margin: 8px 0 8px 0;
    padding: 12px 16px;
    border-left: 3px solid #C2C0B6;
    background: #FAFAF7;
    color: #5F5E5A;
    font-size: 15px;
    line-height: 1.7;
    border-radius: 0 6px 6px 0;
  }}
  .deferrable-body.open {{
    display: block;
  }}

  .unit-break {{
    margin-top: 32px;
  }}
</style>
</head>
<body>

<div class="legend">
  <div class="legend-item">
    <div class="legend-swatch" style="background:#1a1a18;"></div>
    <span><b>Bold</b> = Topic sentence</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#378ADD;"></div>
    <span>Numbered = Parallel ideas</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#C2C0B6; width:4px; border-radius:1px;"></div>
    <span>Indented = Supporting detail</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="border:1px dashed #C2C0B6; background:#FAFAF7;"></div>
    <span>Collapsible = Deferrable detail</span>
  </div>
</div>

{content}

<script>
function toggleDef(id) {{
  var body = document.getElementById('body-' + id);
  var btn  = document.getElementById('btn-' + id);
  if (body.classList.contains('open')) {{
    body.classList.remove('open');
    btn.classList.remove('open');
    btn.querySelector('.label').textContent = 'Expand detail';
  }} else {{
    body.classList.add('open');
    btn.classList.add('open');
    btn.querySelector('.label').textContent = 'Collapse';
  }}
}}
</script>

</body>
</html>"""

# ─────────────────────────────────────────────
# 4. SENTENCE SPLITTING
# ─────────────────────────────────────────────

_ABBREVS = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|al|Fig|fig|Eq|eq|No|no|Vol|vol|pp|approx|ca|cf|ed|eds|est|trans)"

def split_sentences(text: str) -> list[str]:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(rf'({_ABBREVS})\.', r'\1<ABBR_DOT>', text)
    text = re.sub(r'(\d)\.(\d)', r'\1<DEC_DOT>\2', text)
    text = text.replace('...', '<ELLIPSIS>')
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\(])', text)
    sentences = []
    for p in parts:
        p = p.replace('<ABBR_DOT>', '.').replace('<DEC_DOT>', '.').replace('<ELLIPSIS>', '...')
        p = p.strip()
        if p:
            sentences.append(p)
    return sentences


def create_numbered_input(sentences: list[str], start_id: int = 1) -> str:
    return "\n".join(f"[{start_id + i}] {s}" for i, s in enumerate(sentences))

# ─────────────────────────────────────────────
# 5. CHUNKING
# ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def chunk_sentences(sentences: list[str], max_tokens: int = 2500) -> list[list[int]]:
    chunks = []
    current_chunk = []
    current_tokens = 0
    for i, sent in enumerate(sentences):
        sent_tokens = estimate_tokens(sent) + 10
        if current_chunk and (current_tokens + sent_tokens > max_tokens):
            chunks.append(current_chunk)
            current_chunk = [i]
            current_tokens = sent_tokens
        else:
            current_chunk.append(i)
            current_tokens += sent_tokens
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ─────────────────────────────────────────────
# 6. GROQ API CALL
# ─────────────────────────────────────────────

def call_groq(numbered_text: str, api_key: str, model: str) -> str:
    client = Groq(api_key=api_key)
    user_msg = (
        "Analyze the structural role of each sentence below. "
        "Return ONLY the JSON object, nothing else.\n\n"
        f"{numbered_text}"
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg:
                # Parse wait time from error or default to 65s
                wait = 65
                match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
                if match:
                    wait = int(float(match.group(1))) + 5
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            raise e

# ─────────────────────────────────────────────
# 7. LABEL PARSING
# ─────────────────────────────────────────────

def parse_labels(raw: str) -> list[dict]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("sentences", "labels", "results", "output", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        if all(str(k).isdigit() for k in data.keys()):
            return [{"id": int(k), **v} for k, v in data.items()]
    return []


def validate_labels(labels: list[dict], num_sentences: int) -> list[dict]:
    label_map = {}
    for lb in labels:
        sid = lb.get("id")
        if sid is not None:
            label_map[int(sid)] = lb
    validated = []
    for i in range(1, num_sentences + 1):
        if i in label_map:
            lb = label_map[i]
            lb["id"] = i
            lb.setdefault("role", "subordinate")
            lb.setdefault("indent", 0)
            lb.setdefault("group", None)
            lb.setdefault("parent_id", None)
            lb.setdefault("confidence", 0.5)
            validated.append(lb)
        else:
            validated.append({
                "id": i, "role": "subordinate", "indent": 0,
                "group": None, "parent_id": None, "confidence": 0.5,
            })
    return validated

# ─────────────────────────────────────────────
# 8. HTML RENDERING
# ─────────────────────────────────────────────

def render_structread(sentences, labels, conf_threshold=0.6):
    parts = []
    i = 0
    n = len(labels)
    def_counter = 0

    while i < n:
        lb = labels[i]
        sid = lb["id"] - 1
        sent = sentences[sid] if sid < len(sentences) else ""
        role = lb["role"]
        indent = lb.get("indent", 0)
        conf = lb.get("confidence", 1.0)

        if conf < conf_threshold:
            parts.append(f'<div style="margin-bottom:8px;">{sent}</div>')
            i += 1
            continue

        if role == "topic":
            parts.append(f'<div class="s-topic">{sent}</div>')
            i += 1

        elif role == "coordinate":
            group = lb.get("group")
            items = []
            while i < n and labels[i]["role"] == "coordinate" and labels[i].get("group") == group:
                coord_sid = labels[i]["id"] - 1
                coord_sent = sentences[coord_sid] if coord_sid < len(sentences) else ""
                coord_id = labels[i]["id"]
                children = []
                j = i + 1
                while j < n and labels[j].get("parent_id") == coord_id and labels[j]["role"] in ("subordinate", "deferrable"):
                    children.append(labels[j])
                    j += 1
                items.append((coord_sent, children))
                i = j
            indent_px = indent * 32
            parts.append(f'<div class="coord-group" style="margin-left:{indent_px}px;"><ol>')
            for item_sent, children in items:
                parts.append(f'<li>{item_sent}')
                for child in children:
                    child_sid = child["id"] - 1
                    child_sent = sentences[child_sid] if child_sid < len(sentences) else ""
                    if child["role"] == "deferrable":
                        def_counter += 1
                        did = f'd{def_counter}'
                        parts.append(
                            f'<div class="deferrable-wrapper">'
                            f'<button class="deferrable-toggle" id="btn-{did}" onclick="toggleDef(\'{did}\')">'
                            f'<span class="arrow">▶</span> <span class="label">Expand detail</span></button>'
                            f'<div class="deferrable-body" id="body-{did}">{child_sent}</div></div>')
                    else:
                        c_indent = "indent-2" if child.get("indent", 1) >= 2 else ""
                        parts.append(f'<div class="s-subordinate {c_indent}">{child_sent}</div>')
                parts.append('</li>')
            parts.append('</ol></div>')

        elif role == "subordinate":
            cls = "s-subordinate" + (" indent-2" if indent >= 2 else "")
            parts.append(f'<div class="{cls}">{sent}</div>')
            i += 1

        elif role == "deferrable":
            def_counter += 1
            did = f'd{def_counter}'
            parts.append(
                f'<div class="deferrable-wrapper">'
                f'<button class="deferrable-toggle" id="btn-{did}" onclick="toggleDef(\'{did}\')">'
                f'<span class="arrow">▶</span> <span class="label">Expand detail</span></button>'
                f'<div class="deferrable-body" id="body-{did}">{sent}</div></div>')
            i += 1

        elif role == "transition":
            parts.append(f'<div class="s-transition">{sent}</div>')
            i += 1

        else:
            parts.append(f'<div style="margin-bottom:8px;">{sent}</div>')
            i += 1

    return HTML_TEMPLATE.format(content="\n".join(parts))


def render_original(sentences):
    text = " ".join(sentences)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<style>
  body {{
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 17px; line-height: 1.8; color: #2C2C2A;
    background: #FFFFFF; padding: 32px 24px;
    max-width: 720px; margin: 0 auto;
  }}
</style></head>
<body><p>{text}</p></body></html>"""

# ─────────────────────────────────────────────
# 9. PDF EXTRACTION
# ─────────────────────────────────────────────

def extract_pdf_text(uploaded_file):
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        pages = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n\n".join(pages)
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

# ─────────────────────────────────────────────
# 10. STREAMLIT APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="StructRead",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Session state init ──
    if "loaded_text" not in st.session_state:
        st.session_state.loaded_text = ""

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## StructRead")
        st.caption("Structural offloading for reading persistence")
        st.divider()

        api_key = st.text_input(
            "Groq API Key", type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Get one free at console.groq.com",
        )

        model = st.selectbox("Model", [
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "openai/gpt-oss-20b"
        ], index=0)

        st.divider()

        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.05)
        chunk_size = st.slider("Chunk size (tokens)", 1000, 5000, 1200, 500,
                               help="Free tier: keep at 2500. Dev tier: increase to 5000.")

        st.divider()
        st.markdown("#### How it works")
        st.markdown(
            "1. **You paste text** (or upload PDF)\n"
            "2. **LLM labels** each sentence's structural role\n"
            "3. **Same words** are re-rendered in a layout that shows the structure\n\n"
            "No text is added, removed, or changed."
        )

    # ── Main area ──
    st.markdown("# 📐 StructRead")
    st.markdown("*Externalize discourse structure onto the page — same words, new layout.*")

    # ── Input section ──
    st.markdown("### Input")

    input_method = st.radio(
        "Choose input method:",
        ["Paste text", "Upload PDF", "Use sample"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_method == "Paste text":
        typed = st.text_area(
            "Paste your text here", height=250,
            placeholder="Paste a paragraph, section, or chapter…",
            key="paste_area",
        )
        if typed:
            st.session_state.loaded_text = typed

    elif input_method == "Upload PDF":
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")
        if uploaded:
            pdf_text = extract_pdf_text(uploaded)
            if pdf_text:
                st.session_state.loaded_text = pdf_text
                st.success(f"Extracted {len(pdf_text)} characters from PDF.")
                with st.expander("Preview extracted text"):
                    st.text(pdf_text[:2000] + ("…" if len(pdf_text) > 2000 else ""))

    elif input_method == "Use sample":
        st.markdown("A paragraph about sleep deprivation and cognitive performance.")
        if st.button("Load sample text"):
            st.session_state.loaded_text = SAMPLE_TEXT
            st.rerun()

    # ── Show status and always show button ──
    text = st.session_state.loaded_text

    if text:
        st.success(f"✓ Text loaded — {len(text)} characters, ~{estimate_tokens(text)} tokens")
    else:
        st.info("Paste text, upload a PDF, or load the sample to get started.")

    st.divider()

    # Button is ALWAYS visible
    if st.button("🔍 Analyze structure", type="primary", use_container_width=True):
        if not text:
            st.error("No text loaded. Paste, upload, or load the sample first.")
            return
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
            return

        # Step 1: Split sentences
        with st.spinner("Splitting sentences…"):
            sentences = split_sentences(text)
        st.caption(f"{len(sentences)} sentences identified")

        # Step 2: Chunk
        chunks = chunk_sentences(sentences, max_tokens=chunk_size)
        total_chunks = len(chunks)

        if total_chunks > 1:
            est_time = total_chunks * 15
            st.info(
                f"📦 Text split into **{total_chunks} chunks** for rate limits. "
                f"Estimated time: **~{math.ceil(est_time / 60)} min**. "
            )

        # Step 3: Process
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            if total_chunks == 1:
                status_text.text("Analyzing discourse structure…")
                numbered = create_numbered_input(sentences)
                raw_response = call_groq(numbered, api_key, model)
                all_labels = parse_labels(raw_response)
                progress_bar.progress(1.0)
                status_text.text("✓ Analysis complete.")
            else:
                all_labels = process_all_chunks(
                    sentences, chunks, api_key, model,
                    progress_bar, status_text
                )
        except Exception as e:
            st.error(f"Groq API error: {e}")
            return

        if not all_labels:
            st.error("Failed to parse LLM response.")
            return

        # Step 4: Render
        labels = validate_labels(all_labels, len(sentences))
        html_struct = render_structread(sentences, labels, conf_threshold)
        html_orig = render_original(sentences)

        # ── Tabbed display ──
        st.divider()
        tab_struct, tab_orig = st.tabs(["📐 StructRead", "📄 Original"])

        with tab_struct:
            st.components.v1.html(html_struct, height=800, scrolling=True)

        with tab_orig:
            st.components.v1.html(html_orig, height=800, scrolling=True)

        # ── Label inspection ──
        with st.expander("View structural labels"):
            role_counts = {}
            for lb in labels:
                role_counts[lb["role"]] = role_counts.get(lb["role"], 0) + 1
            cols = st.columns(len(role_counts))
            for i, (role, count) in enumerate(role_counts.items()):
                cols[i].metric(role, count)
            st.divider()
            table_data = []
            for lb in labels:
                sid = lb["id"] - 1
                s = sentences[sid] if sid < len(sentences) else ""
                table_data.append({
                    "id": lb["id"],
                    "sentence": (s[:80] + "…") if len(s) > 80 else s,
                    "role": lb["role"],
                    "indent": lb.get("indent", 0),
                    "group": lb.get("group") or "—",
                    "parent": lb.get("parent_id") or "—",
                    "conf": f"{lb.get('confidence', 0):.0%}",
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
