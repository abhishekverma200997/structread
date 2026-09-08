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

  /* ── Legend ── */
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

  /* ── Structural roles ── */
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

  /* ── Deferrable: collapsible ── */
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

  /* ── Whitespace between structural units ── */
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

# Common abbreviations that shouldn't trigger a split
_ABBREVS = r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|al|Fig|fig|Eq|eq|No|no|Vol|vol|pp|approx|ca|cf|ed|eds|est|trans)"

def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles abbreviations and decimal numbers."""
    # Normalize whitespace but preserve paragraph structure
    text = re.sub(r'[ \t]+', ' ', text)

    # Protect abbreviations
    text = re.sub(rf'({_ABBREVS})\.', r'\1<ABBR_DOT>', text)
    # Protect decimal numbers (e.g. 0.05)
    text = re.sub(r'(\d)\.(\d)', r'\1<DEC_DOT>\2', text)
    # Protect ellipsis
    text = text.replace('...', '<ELLIPSIS>')

    # Split on sentence-ending punctuation followed by space + uppercase or end
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'\(])', text)

    # Restore protected tokens
    sentences = []
    for p in parts:
        p = p.replace('<ABBR_DOT>', '.')
        p = p.replace('<DEC_DOT>', '.')
        p = p.replace('<ELLIPSIS>', '...')
        p = p.strip()
        if p:
            sentences.append(p)

    return sentences


def create_numbered_input(sentences: list[str]) -> str:
    """Format sentences as numbered input for the LLM."""
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences))

# ─────────────────────────────────────────────
# 5. GROQ API CALL
# ─────────────────────────────────────────────

def call_groq(numbered_text: str, api_key: str, model: str) -> str:
    """Send sentences to Groq and get structural labels back."""
    client = Groq(api_key=api_key)

    user_msg = (
        "Analyze the structural role of each sentence below. "
        "Return ONLY the JSON object, nothing else.\n\n"
        f"{numbered_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=8192,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# 6. LABEL PARSING
# ─────────────────────────────────────────────

def parse_labels(raw: str) -> list[dict]:
    """Parse the LLM JSON response into a list of label dicts."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from possible markdown wrapping
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return []

    # The prompt asks for {"sentences": [...]}, but handle variations
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("sentences", "labels", "results", "output", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # If it's a dict of dicts keyed by id
        if all(k.isdigit() for k in data.keys()):
            return [{"id": int(k), **v} for k, v in data.items()]
    return []


def validate_labels(labels: list[dict], num_sentences: int) -> list[dict]:
    """Ensure every sentence has a label; fill gaps with safe defaults."""
    label_map = {}
    for lb in labels:
        sid = lb.get("id")
        if sid is not None:
            label_map[int(sid)] = lb

    validated = []
    for i in range(1, num_sentences + 1):
        if i in label_map:
            lb = label_map[i]
            # Normalize fields
            lb["id"] = i
            lb.setdefault("role", "subordinate")
            lb.setdefault("indent", 0)
            lb.setdefault("group", None)
            lb.setdefault("parent_id", None)
            lb.setdefault("confidence", 0.5)
            validated.append(lb)
        else:
            # Missing label — safe default
            validated.append({
                "id": i,
                "role": "subordinate",
                "indent": 0,
                "group": None,
                "parent_id": None,
                "confidence": 0.5,
            })
    return validated

# ─────────────────────────────────────────────
# 7. HTML RENDERING — the rendering engine
# ─────────────────────────────────────────────

def render_structread(sentences: list[str], labels: list[dict], conf_threshold: float = 0.6) -> str:
    """
    Convert sentences + structural labels into the StructRead HTML layout.

    Visual encoding (from StructRead proposal):
      - List markers     → Coordination
      - Indentation      → Hierarchical subordination
      - Left vertical rule → Deferrable detail
      - Inter-block whitespace → Structural unit boundary
    """
    parts = []
    i = 0
    n = len(labels)
    def_counter = 0

    while i < n:
        lb = labels[i]
        sid = lb["id"] - 1  # 0-indexed
        sent = sentences[sid] if sid < len(sentences) else ""
        role = lb["role"]
        indent = lb.get("indent", 0)
        conf = lb.get("confidence", 1.0)

        # Low confidence → render as plain text
        if conf < conf_threshold:
            parts.append(f'<div style="margin-bottom:8px;">{sent}</div>')
            i += 1
            continue

        # ── TOPIC ──
        if role == "topic":
            parts.append(f'<div class="s-topic">{sent}</div>')
            i += 1

        # ── COORDINATE ──
        elif role == "coordinate":
            group = lb.get("group")
            # Collect consecutive coordinate sentences in the same group
            items = []
            while i < n and labels[i]["role"] == "coordinate" and labels[i].get("group") == group:
                coord_sid = labels[i]["id"] - 1
                coord_sent = sentences[coord_sid] if coord_sid < len(sentences) else ""

                # Gather immediate children (subordinate/deferrable with this parent)
                coord_id = labels[i]["id"]
                children = []
                j = i + 1
                while j < n and labels[j].get("parent_id") == coord_id and labels[j]["role"] in ("subordinate", "deferrable"):
                    children.append(labels[j])
                    j += 1

                items.append((coord_sent, children))
                i = j  # skip past children

            # Render the group as an ordered list
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
                            f'<div class="deferrable-body" id="body-{did}">{child_sent}</div>'
                            f'</div>'
                        )
                    else:
                        c_indent = "indent-2" if child.get("indent", 1) >= 2 else ""
                        parts.append(f'<div class="s-subordinate {c_indent}">{child_sent}</div>')
                parts.append('</li>')
            parts.append('</ol></div>')

        # ── SUBORDINATE ──
        elif role == "subordinate":
            cls = "s-subordinate"
            if indent >= 2:
                cls += " indent-2"
            parts.append(f'<div class="{cls}">{sent}</div>')
            i += 1

        # ── DEFERRABLE ──
        elif role == "deferrable":
            def_counter += 1
            did = f'd{def_counter}'
            parts.append(
                f'<div class="deferrable-wrapper">'
                f'<button class="deferrable-toggle" id="btn-{did}" onclick="toggleDef(\'{did}\')">'
                f'<span class="arrow">▶</span> <span class="label">Expand detail</span></button>'
                f'<div class="deferrable-body" id="body-{did}">{sent}</div>'
                f'</div>'
            )
            i += 1

        # ── TRANSITION ──
        elif role == "transition":
            parts.append(f'<div class="s-transition">{sent}</div>')
            i += 1

        else:
            parts.append(f'<div style="margin-bottom:8px;">{sent}</div>')
            i += 1

    content = "\n".join(parts)
    return HTML_TEMPLATE.format(content=content)


def render_original(sentences: list[str]) -> str:
    """Render the original text as a plain wall of prose for comparison."""
    text = " ".join(sentences)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<style>
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
</style></head>
<body><p>{text}</p></body></html>"""

# ─────────────────────────────────────────────
# 8. PDF EXTRACTION
# ─────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """Extract text from a PDF upload."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n\n".join(pages)
    except ImportError:
        st.error("PyPDF2 not installed. Run: pip install PyPDF2")
        return ""
    except Exception as e:
        st.error(f"PDF extraction failed: {e}")
        return ""

# ─────────────────────────────────────────────
# 9. STREAMLIT APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="StructRead",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## StructRead")
        st.caption("Structural offloading for reading persistence")
        st.divider()

        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.environ.get("GROQ_API_KEY", ""),
            help="Get one free at console.groq.com",
        )

        model = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
            help="70B models produce better structural analysis",
        )

        st.divider()

        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Labels below this confidence render as plain text",
        )

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

    # Input tabs
    tab_paste, tab_upload, tab_sample = st.tabs(["Paste text", "Upload PDF", "Use sample"])

    text = ""
    with tab_paste:
        text_input = st.text_area(
            "Paste your text here",
            height=250,
            placeholder="Paste a paragraph, section, or chapter…",
        )
        if text_input:
            text = text_input

    with tab_upload:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded:
            text = extract_pdf_text(uploaded)
            if text:
                st.success(f"Extracted {len(text)} characters from PDF.")
                with st.expander("Preview extracted text"):
                    st.text(text[:2000] + ("…" if len(text) > 2000 else ""))

    with tab_sample:
        st.markdown("A paragraph about sleep deprivation and cognitive performance.")
        if st.button("Load sample text"):
            text = SAMPLE_TEXT
            st.success("Sample text loaded.")

    if not text:
        st.info("Paste text, upload a PDF, or load the sample to get started.")
        return

    # ── Process ──
    if st.button("Analyze structure", type="primary", use_container_width=True):
        if not api_key:
            st.error("Enter your Groq API key in the sidebar.")
            return

        # Step 1: Split sentences
        with st.spinner("Splitting sentences…"):
            sentences = split_sentences(text)
            numbered = create_numbered_input(sentences)

        st.caption(f"{len(sentences)} sentences identified")

        # Step 2: Call LLM for structural labels
        with st.spinner("LLM is analyzing discourse structure…"):
            try:
                raw_response = call_groq(numbered, api_key, model)
            except Exception as e:
                st.error(f"Groq API error: {e}")
                return

        # Step 3: Parse labels
        labels = parse_labels(raw_response)
        if not labels:
            st.error("Failed to parse LLM response. Raw output:")
            st.code(raw_response)
            return

        labels = validate_labels(labels, len(sentences))

        # Step 4: Render
        html_struct = render_structread(sentences, labels, conf_threshold)
        html_orig = render_original(sentences)

        # ── Side-by-side display ──
        st.divider()
        col_orig, col_struct = st.columns(2)

        with col_orig:
            st.markdown("### Original text")
            st.components.v1.html(html_orig, height=600, scrolling=True)

        with col_struct:
            st.markdown("### StructRead")
            st.components.v1.html(html_struct, height=600, scrolling=True)

        # ── Label inspection ──
        with st.expander("View structural labels (raw)"):
            # Summary stats
            role_counts = {}
            for lb in labels:
                r = lb["role"]
                role_counts[r] = role_counts.get(r, 0) + 1

            cols = st.columns(len(role_counts))
            for i, (role, count) in enumerate(role_counts.items()):
                cols[i].metric(role, count)

            st.divider()

            # Per-sentence table
            table_data = []
            for lb in labels:
                sid = lb["id"] - 1
                table_data.append({
                    "id": lb["id"],
                    "sentence": (sentences[sid][:80] + "…") if sid < len(sentences) and len(sentences[sid]) > 80 else (sentences[sid] if sid < len(sentences) else ""),
                    "role": lb["role"],
                    "indent": lb.get("indent", 0),
                    "group": lb.get("group", "—"),
                    "parent": lb.get("parent_id", "—"),
                    "confidence": f"{lb.get('confidence', 0):.0%}",
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

        # ── Raw LLM response ──
        with st.expander("Raw LLM JSON response"):
            st.code(raw_response, language="json")

        # Store in session for re-rendering without re-calling API
        st.session_state["sentences"] = sentences
        st.session_state["labels"] = labels


if __name__ == "__main__":
    main()
