"""
StructRead — Structural Offloading for Reading Persistence
Proof-of-concept prototype

Takes any text → LLM labels discourse structure → renders the same words
in a layout that externalizes that structure.
"""

import streamlit as st
import json
import re
import os
import time
import math
import html
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

# Words whose trailing "." is not a sentence end (compared lowercase, without the final dot)
_ABBREV_WORDS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "al", "fig", "figs", "eq", "eqs",
    "no", "nos", "vol", "vols", "pp", "p", "approx", "ca", "cf", "ed", "eds", "est", "trans", "dept",
    "univ", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "sec", "ch", "chap", "ref", "refs", "viz", "resp", "ibid", "e.g", "i.e",
    "a.m", "p.m", "ph.d", "u.s", "u.k",
}
# Sentence-final punctuation run, plus any closing quotes/brackets after it
_SENT_END_RE = re.compile(r'(?:\.{3}|…|[.!?。！？])+["\'”’)\]]*')


def _is_sentence_start(s: str) -> bool:
    s = s.lstrip('"\'“‘([')
    if not s:
        return False
    # Capital (incl. accented), digit ("3D printing"), or camel-case word ("iPhone", "eBay")
    return s[0].isupper() or s[0].isdigit() or bool(re.match(r'[a-z]+[A-Z]', s))


def split_sentences(text: str) -> list[str]:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'(\w)-\n([a-z])', r'\1\2', text)  # re-join words hyphenated across PDF lines
    sentences, start = [], 0
    for m in _SENT_END_RE.finditer(text):
        end, punct = m.end(), m.group()
        if punct[0] not in "。！？":  # CJK full stops always end a sentence
            ws = re.match(r'\s+', text[end:])
            if not ws or not _is_sentence_start(text[end + ws.end():]):
                continue
            if punct.startswith(("...", "…")):
                continue  # ellipsis: keep the thought together
            if punct.rstrip('"\'”’)]') == ".":
                before = text[start:m.start()].split()
                word = before[-1].lstrip('"\'“‘([').lower() if before else ""
                next_char = text[end + ws.end():][:1]
                if (
                    (word in _ABBREV_WORDS and not (word in ("no", "nos") and not next_char.isdigit()))
                    or re.fullmatch(r'[^\W\d_]', word)                        # initial: "J. K. Rowling"
                    or re.fullmatch(r'(?:[^\W\d_]\.)+[^\W\d_]', word)          # "U.S.", "e.g."
                    or (len(before) == 1 and re.fullmatch(r'\d+|[ivxlc]+', word))  # list marker: "1."
                ):
                    continue
        s = text[start:end].strip()
        if s:
            sentences.append(s)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


# Sections that are lists/navigation rather than prose — rendered as-is, never sent to the LLM
_PASSTHROUGH = (r"references|bibliography|works cited|literature cited|sources|citations|"
                r"further reading|suggested reading|notes|endnotes|footnotes|"
                r"table of contents|contents|list of (?:figures|tables|illustrations|abbreviations)|"
                r"abbreviations|index|glossary")
# Back-matter prose headings that end a passthrough section (e.g. an appendix after the references)
_BACK_PROSE = r"appendix(?:\s+\w+)?|appendices|acknowledge?ments?|about the authors?|afterword"
_HEAD_NUM = r"(?:(?:[IVXLC]+[.)]?|\d+(?:\.\d+)*[.)]?|[A-Z][.)])\s+)?"
_PASS_RE = re.compile(rf"^{_HEAD_NUM}({_PASSTHROUGH})\s*:?$", re.I)
_BACK_PROSE_RE = re.compile(rf"^{_HEAD_NUM}(?:{_BACK_PROSE})\s*:?$", re.I)
_PAGE_NUM_RE = re.compile(r"(?:\.{2,}|\s)\s*(?:\d+|[ivxlcIVXLC]+)\s*$")


def split_passthrough_sections(text: str) -> list[tuple[str, str, str]]:
    """Split text into ("prose", "", body) and ("verbatim", heading, body) segments."""
    lines = text.split("\n")
    segments, buf = [], []

    def flush_prose():
        if "\n".join(buf).strip():
            segments.append(("prose", "", "\n".join(buf)))
        buf.clear()

    i = 0
    while i < len(lines):
        m = _PASS_RE.match(lines[i].strip())
        if not m:
            buf.append(lines[i])
            i += 1
            continue
        flush_prose()
        kind = m.group(1).lower()
        j = i + 1
        if "contents" in kind:
            # A table of contents ends at the first prose-like line (long, no page number)
            while j < len(lines):
                t = lines[j].strip()
                if len(t) > 60 and not _PAGE_NUM_RE.search(t):
                    break
                j += 1
            # Give trailing unnumbered lines (e.g. the first real heading) back to the prose
            if any(_PAGE_NUM_RE.search(l.strip()) for l in lines[i + 1:j]):
                while j > i + 1 and not _PAGE_NUM_RE.search(lines[j - 1].strip()):
                    j -= 1
        else:
            # Back-matter lists run until another known section heading (or the end)
            while j < len(lines):
                t = lines[j].strip()
                other = _PASS_RE.match(t)
                if _BACK_PROSE_RE.match(t) or (other and other.group(1).lower() != kind):
                    break
                j += 1
        segments.append(("verbatim", lines[i].strip(), "\n".join(lines[i + 1:j])))
        i = j
    flush_prose()
    return segments


def render_verbatim(title: str, body: str) -> str:
    return (
        '<div style="margin:32px 0 16px 0;">'
        '<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;'
        'font-size:12px;color:#888780;margin-bottom:6px;">Shown as-is — not restructured</div>'
        f'<div style="font-weight:600;color:#1a1a18;margin-bottom:8px;">{html.escape(title)}</div>'
        '<div style="white-space:pre-wrap;font-size:15px;line-height:1.6;color:#3d3d3b;">'
        f'{html.escape(body.strip())}</div></div>'
    )


_HEADING_WORDS_RE = re.compile(
    r"^(?:abstract|introduction|conclusions?|discussion|results|methods?|methodology|background|"
    r"related work|summary|preface|foreword|prologue|epilogue|afterword|acknowledge?ments?|"
    r"appendix(?:\s+\w+)?|(?:chapter|part)\s+[\w.]+(?:\s*[:.—–-]\s*.*)?)\s*:?$", re.I)
_NUMBERED_HEAD_RE = re.compile(r"^(?:[IVXLC]+[.)]|\d{1,2}(?:\.\d{1,2})*[.)]?|[A-Z][.)])\s+(\S.*)$")
_CAPTION_RE = re.compile(
    # "Fig. 2." / "Table 3:" / a bare "TABLE I" line — but not prose like "Figure 3 shows…"
    r"^(?:fig\.|figure|table|algorithm|listing)\s*[\dIVX]+[a-z]?(?:\s*[.:—–-]|[ \t]*$)", re.I | re.M)
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs", "with"}


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _HEADING_WORDS_RE.match(s):
        return True
    if s[-1] in ".,;:!?" or not s[0].isalnum() or _letter_ratio(s) < 0.75 or _CAPTION_RE.match(s):
        return False  # sentence end, an equation line like "= DKL(q || p)", or a "TABLE I" caption
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters) and len(s.split()) <= 10:
        return True  # ALL-CAPS line: "I. INTRODUCTION", "RELATED WORK"
    m = _NUMBERED_HEAD_RE.match(s)
    if m:
        words = [w for w in m.group(1).split() if w.lower() not in _SMALL_WORDS]
        if len(words) == 1 and len(words[0]) < 5:
            return False  # list fragment like "3) We"
        if re.search(r":\s+\S", m.group(1)):
            return False  # run-in heading that continues as prose: "1) Training CNNs: First, we train"
        if 1 <= len(words) <= 10 and m.group(1)[0].isupper():
            capitalized = sum(1 for w in words if w[0].isupper() or w[0].isdigit())
            return capitalized / len(words) >= 0.6  # "2.1 Data Collection", not "1. Smith said that"
    return False


def _letter_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 1.0


def _segment_prose(body: str) -> list[tuple[str, str, str]]:
    """Pull headings, captions and table/equation blocks out of prose so they aren't read as sentences."""
    segments, buf = [], []

    def flush():
        if "\n".join(buf).strip():
            segments.append(("prose", "", "\n".join(buf)))
        buf.clear()

    def table_like(p):
        # Short lines with no sentence endings: table cells / column headers / wrapped caption text
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        return all(len(l) <= 60 and not l.endswith(".") for l in lines) and not (
            _NUMBERED_HEAD_RE.match(lines[0]) and _is_heading(lines[0]))

    in_figure = False  # after a "Table 2" / "Fig. 3" caption, absorb the table body that follows
    for para in re.split(r'\n\s*\n', body):
        p = para.strip()
        if not p:
            continue
        m = ASSET_MARKER_RE.fullmatch(p)
        if m:  # image / chart / table snapshot captured from the PDF
            flush()
            segments.append(("asset", m.group(1), ""))
            in_figure = False
            continue
        if _is_pasted_table(p):
            flush()
            segments.append(("table", "", p))
            in_figure = False
            continue
        if _CAPTION_RE.match(p):
            flush()
            segments.append(("figure", "", p))
            in_figure = True
            continue
        if in_figure and (_letter_ratio(p) < 0.5 or table_like(p) or p.isupper()):
            segments[-1] = ("figure", "", segments[-1][2] + "\n" + p)
            continue
        in_figure = False
        if _letter_ratio(p) < 0.5:
            flush()
            segments.append(("figure", "", p))  # table cells, equation
            continue
        for line in para.split("\n"):
            s = line.strip()
            if _is_heading(line):
                last = segments[-1] if segments else None
                if (last and last[0] == "heading" and not "".join(buf).strip() and s.isupper()
                        and last[1].isupper() and not _NUMBERED_HEAD_RE.match(s)):
                    # ALL-CAPS heading wrapped onto a second line (or block), nothing in between
                    segments[-1] = ("heading", f"{last[1]} {s}", "")
                else:
                    flush()
                    segments.append(("heading", s, ""))
            else:
                buf.append(line)
        buf.append("")  # keep the paragraph break
    flush()
    return segments


_PIPE_SEPARATOR_RE = re.compile(r'^\|?[\s:|-]+\|?$')


def _is_pasted_table(p: str) -> bool:
    """Tab-separated rows (copied from a spreadsheet/web page) or a Markdown | pipe | table."""
    lines = [l for l in p.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    if all("\t" in l for l in lines):
        return True
    return all(l.strip().startswith("|") and l.strip().endswith("|") for l in lines)


def _table_rows(p: str) -> list[list[str]]:
    rows = []
    for l in p.split("\n"):
        s = l.strip()
        if not s or _PIPE_SEPARATOR_RE.fullmatch(s) and "-" in s:
            continue  # blank line or Markdown header separator |---|---|
        cells = s.strip("|").split("|") if s.startswith("|") else l.split("\t")
        rows.append([c.strip() for c in cells])
    return rows


def render_table(p: str) -> str:
    rows = _table_rows(p)
    if not rows:
        return render_figure(p)
    width = max(len(r) for r in rows)
    cell = 'padding:6px 10px;border-bottom:1px solid #EEEDEA;text-align:left;vertical-align:top;'
    head = "".join(f'<th style="{cell}background:#F6F5F0;font-weight:600;position:sticky;top:0;">'
                   f'{html.escape(c)}</th>' for c in rows[0] + [""] * (width - len(rows[0])))
    body = "".join(
        f'<tr style="background:{"#FFFFFF" if k % 2 == 0 else "#FAFAF7"};">'
        + "".join(f'<td style="{cell}">{html.escape(c)}</td>' for c in r + [""] * (width - len(r)))
        + "</tr>"
        for k, r in enumerate(rows[1:]))
    return ('<div style="overflow-x:auto;margin:16px 0;font-family:-apple-system,BlinkMacSystemFont,'
            '\'Segoe UI\',sans-serif;font-size:14px;">'
            f'<table style="border-collapse:collapse;min-width:50%;"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def render_asset(data_uri) -> str:
    if not data_uri:
        return '<span style="color:#888780;font-size:13px;">[figure not available]</span>'
    return (f'<img src="{data_uri}" alt="Figure from the PDF" style="max-width:100%;height:auto;'
            'display:block;margin:0 auto;border:1px solid #EEEDEA;border-radius:4px;">')


def inject_assets(page_html: str, assets: dict) -> str:
    """Swap [[SR-ASSET:id]] markers for the captured images."""
    return ASSET_MARKER_RE.sub(lambda m: render_asset((assets or {}).get(m.group(1))), page_html)


def segment_text(text: str) -> list[tuple[str, str, str]]:
    """(kind, title, body) segments; kind is prose | verbatim | heading | figure | table | asset.
    Only prose goes to the LLM."""
    out = []
    for kind, title, body in split_passthrough_sections(text):
        out.extend(_segment_prose(body) if kind == "prose" else [(kind, title, body)])
    return out


def render_heading(title: str) -> str:
    return ('<div style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;'
            'font-size:19px;font-weight:700;color:#1a1a18;margin:36px 0 12px 0;">'
            f'{html.escape(title)}</div>')


def render_figure(body: str) -> str:
    return ('<div style="white-space:pre-wrap;font-size:14px;line-height:1.6;color:#5F5E5A;'
            'margin:14px 0;padding:10px 14px;background:#FAFAF7;border-radius:6px;">'
            f'{html.escape(body)}</div>')


@st.cache_data(show_spinner=False)
def prepare_text(text: str):
    """Split text into prose sentences for the LLM, plus as-is HTML blocks keyed by sentence position."""
    sentences, inserts, kept_titles = [], {}, []
    for kind, title, body in segment_text(text):
        if kind == "prose":
            sentences.extend(split_sentences(body))
            continue
        if kind == "verbatim":
            block = render_verbatim(title, body)
            kept_titles.append(title)
        elif kind == "heading":
            block = render_heading(title)
        elif kind == "asset":
            block = f'<div style="margin:18px 0;">[[SR-ASSET:{title}]]</div>'  # filled by inject_assets
        elif kind == "table":
            block = render_table(body)
        else:
            block = render_figure(body)
        inserts.setdefault(len(sentences), []).append(block)
    return sentences, inserts, kept_titles


def estimate_minutes(total_chunks: int) -> int:
    # ~15s per Groq call + the 65s rate-limit pause between chunks (see process_all_chunks)
    return math.ceil((total_chunks * 15 + max(total_chunks - 1, 0) * 65) / 60)


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

class RateLimitExhausted(Exception):
    """Groq asked us to wait too long (e.g. a daily token limit) — stop instead of sleeping."""
    def __init__(self, wait_seconds, message):
        super().__init__(message)
        self.wait_seconds = wait_seconds


def parse_retry_after(error_msg: str):
    """Seconds from Groq's 'Please try again in 7m12.48s' (also handles h / s / ms); None if absent."""
    m = re.search(r'try again in\s+((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)', error_msg)
    if not m:
        return None
    unit_seconds = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    return sum(float(v) * unit_seconds[u] for v, u in re.findall(r'(\d+(?:\.\d+)?)(ms|h|m|s)', m.group(1)))


def call_groq(numbered_text: str, api_key: str, model: str, on_wait=None) -> str:
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
                wait = parse_retry_after(error_msg) or 60
                daily = "per day" in error_msg or "(TPD)" in error_msg or "(RPD)" in error_msg
                if daily or wait > 180 or attempt == max_retries - 1:
                    raise RateLimitExhausted(wait, error_msg)
                for remaining in range(math.ceil(wait) + 5, 0, -1):
                    if on_wait:
                        on_wait(remaining)
                    time.sleep(1)
                continue
            raise e


def process_all_chunks(sentences, chunks, api_key, model, progress_bar, status_text):
    """Label every chunk. A chunk that fails is skipped (its sentences render as plain text)
    instead of throwing away the whole run.

    Returns (labels, failed, stop_message): failed is a list of (first_id, last_id, reason).
    """
    all_labels, failed, stop_message = [], [], None
    total_chunks = len(chunks)

    def on_wait(remaining):
        status_text.text(f"Groq rate limit hit — retrying in {remaining}s…")

    for chunk_idx, chunk_indices in enumerate(chunks):
        chunk_sents = [sentences[i] for i in chunk_indices]
        start_id = chunk_indices[0] + 1
        numbered = create_numbered_input(chunk_sents, start_id=start_id)
        if chunk_idx > 0:
            # Unnumbered so the LLM has nothing to label; ids stay those of the current chunk
            context_sents = [sentences[i] for i in chunks[chunk_idx - 1][-3:]]
            numbered = (
                "[CONTEXT — do not label these sentences, they are from the previous section for reference only]\n"
                + "\n".join(context_sents)
                + "\n\n[LABEL ONLY THE NUMBERED SENTENCES BELOW]\n"
                + numbered
            )
        status_text.text(f"Analyzing chunk {chunk_idx + 1} of {total_chunks} "
                         f"({len(chunk_sents)} sentences)…")
        progress_bar.progress((chunk_idx) / total_chunks)
        chunk_ids = {i + 1 for i in chunk_indices}
        first_id, last_id = min(chunk_ids), max(chunk_ids)
        best, reason = [], None
        for attempt in range(2):  # one retry for an incomplete / unparseable response
            try:
                raw = call_groq(numbered, api_key, model, on_wait=on_wait)
            except RateLimitExhausted as e:
                mins = math.ceil(e.wait_seconds / 60)
                stop_message = (f"Groq's rate limit was reached after {chunk_idx} of {total_chunks} chunks "
                                f"(Groq says to try again in ~{mins} min). The rest is shown as plain text.")
                failed.append((first_id, len(sentences), "rate limit reached"))
                break
            except Exception as e:
                if chunk_idx == 0:
                    raise  # first call failing is a setup problem (key, model, network) — report it
                reason = f"API error: {str(e)[:120]}"
                break
            # Drop any labels outside this chunk so they can't overwrite earlier chunks' labels
            chunk_labels = [lb for lb in parse_labels(raw)
                            if isinstance(lb, dict) and _to_int(lb.get("id")) in chunk_ids]
            if len(chunk_labels) > len(best):
                best = chunk_labels
            if len(best) >= 0.9 * len(chunk_ids):
                reason = None
                break
            reason = "Groq returned an incomplete response"
            status_text.text(f"Chunk {chunk_idx + 1}: incomplete response — retrying…")
        all_labels.extend(best)
        if stop_message:
            break
        if reason and len(best) < 0.5 * len(chunk_ids):
            failed.append((first_id, last_id, reason))
        if chunk_idx < total_chunks - 1:
            wait_seconds = 65
            chunks_left = total_chunks - chunk_idx - 1
            for remaining in range(wait_seconds, 0, -1):
                overall = remaining + chunks_left * 15 + (chunks_left - 1) * 65
                status_text.text(f"✓ Chunk {chunk_idx + 1} of {total_chunks} done. "
                                 f"Waiting {remaining}s for rate limit… "
                                 f"(~{math.ceil(overall / 60)} min left overall)")
                time.sleep(1)
    progress_bar.progress(1.0)
    if stop_message:
        status_text.text("Stopped early — Groq rate limit reached.")
    else:
        status_text.text(f"✓ All {total_chunks} chunk{'s' if total_chunks > 1 else ''} processed.")
    return all_labels, failed, stop_message

# ─────────────────────────────────────────────
# 7. LABEL PARSING
# ─────────────────────────────────────────────

def parse_labels(raw: str) -> list[dict]:
    """Never raises: malformed or truncated output yields whatever complete labels can be salvaged."""
    if not raw:
        return []
    data = None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                data = None
    if data is None:
        # Truncated / broken JSON: keep every complete {...} label object
        salvaged = []
        for m in re.finditer(r'\{[^{}]*\}', raw):
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "id" in obj:
                salvaged.append(obj)
        return salvaged
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("sentences", "labels", "results", "output", "data"):
            if key in data and isinstance(data[key], list):
                return [d for d in data[key] if isinstance(d, dict)]
        if data and all(str(k).isdigit() and isinstance(v, dict) for k, v in data.items()):
            return [{**v, "id": int(k)} for k, v in data.items()]
    return []


_ROLES = {"topic", "coordinate", "subordinate", "deferrable", "transition"}


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_confidence(v) -> float:
    try:
        c = float(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return 0.5
    if 1 < c <= 100:
        c /= 100  # "92" or "92%"
    return min(max(c, 0.0), 1.0)


def validate_labels(labels: list[dict], num_sentences: int) -> list[dict]:
    """Normalize every field so the renderer can trust types and ranges."""
    label_map = {}
    for lb in labels:
        if not isinstance(lb, dict):
            continue
        sid = _to_int(lb.get("id"))
        if sid is not None and 1 <= sid <= num_sentences:
            label_map[sid] = lb
    validated = []
    for i in range(1, num_sentences + 1):
        lb = label_map.get(i)
        if lb is None:
            validated.append({
                "id": i, "role": "subordinate", "indent": 0,
                "group": None, "parent_id": None, "confidence": 0.5,
            })
            continue
        role = str(lb.get("role") or "").strip().lower()
        if role not in _ROLES:
            role = "subordinate"
        # Hard ceiling: never render deeper than indent 2
        indent = min(max(_to_int(lb.get("indent")) or 0, 0), 2)
        group = lb.get("group")
        group = str(group).strip() or None if (role == "coordinate" and group is not None) else None
        parent = _to_int(lb.get("parent_id"))
        if parent is None or not (1 <= parent <= num_sentences) or parent == i:
            parent = None
        validated.append({
            "id": i, "role": role, "indent": indent, "group": group,
            "parent_id": parent, "confidence": _to_confidence(lb.get("confidence", 0.5)),
        })
    return validated

# ─────────────────────────────────────────────
# 8. HTML RENDERING
# ─────────────────────────────────────────────

def render_structread(sentences, labels, conf_threshold=0.6, inserts=None):
    # inserts: {sentence index: [html, ...]} — as-is sections placed before that sentence
    inserts = inserts or {}
    sentences = [html.escape(s) for s in sentences]  # source text is never trusted as HTML
    parts = []
    i = 0
    n = len(labels)
    def_counter = 0

    while i < n:
        parts.extend(inserts.get(i, []))
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
            group_start = i
            # A group never spans an as-is section
            while (i < n and labels[i]["role"] == "coordinate" and labels[i].get("group") == group
                   and (i == group_start or i not in inserts)):
                coord_sid = labels[i]["id"] - 1
                coord_sent = sentences[coord_sid] if coord_sid < len(sentences) else ""
                coord_id = labels[i]["id"]
                children = []
                j = i + 1
                while (j < n and j not in inserts and labels[j].get("parent_id") == coord_id
                       and labels[j]["role"] in ("subordinate", "deferrable")):
                    children.append(labels[j])
                    j += 1
                items.append((coord_sent, children))
                i = j
            indent_px = indent * 32
            # A lone "parallel" item (siblings not adjacent, or split by a heading/table) isn't a list
            is_list = len(items) > 1
            if is_list:
                parts.append(f'<div class="coord-group" style="margin-left:{indent_px}px;"><ol>')
            for item_sent, children in items:
                parts.append(f'<li>{item_sent}' if is_list
                             else f'<div style="margin:0 0 8px {indent_px}px;">{item_sent}')
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
                parts.append('</li>' if is_list else '</div>')
            if is_list:
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

    parts.extend(inserts.get(n, []))
    return HTML_TEMPLATE.format(content="\n".join(parts))


def render_paragraphs(text):
    paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', text)]
    body = "\n".join(f'<p style="margin-bottom:1.2em;">{html.escape(p)}</p>' for p in paragraphs if p)
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
<body>{body}</body></html>"""

# ─────────────────────────────────────────────
# 9. PDF EXTRACTION
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading OCR engine…")
def get_ocr_engine():
    from rapidocr import RapidOCR
    return RapidOCR()


def ocr_page(page):
    """Render one page to an image and read it with OCR (for scanned / image-only pages)."""
    png = page.get_pixmap(dpi=200).tobytes("png")
    result = get_ocr_engine()(png)
    text = ""
    for line in (result.txts or []):
        line = line.strip()
        if text.endswith("-"):
            text = text[:-1] + line  # re-join words hyphenated across lines
        else:
            text = f"{text}\n{line}" if text else line  # keep lines so headings stay detectable
    return text


ASSET_MARKER_RE = re.compile(r'\[\[SR-ASSET:([\w-]+)\]\]')
_TABLE_CAPTION_RE = re.compile(r"^table\s*[\dIVX]+[a-z]?\b", re.I)


def _is_prose_block(text: str) -> bool:
    t = " ".join(text.split())
    return len(t) > 200 and _letter_ratio(t) > 0.7 and ". " in t


def _is_table_block(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    return (_letter_ratio(text) < 0.5 or text.strip().isupper()
            or all(len(l) <= 60 and not l.endswith(".") for l in lines))


def _merge_rects(rects, gap=12):
    rects = [r for r in rects]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                grown = rects[i] + (-gap, -gap, gap, gap)
                if grown.intersects(rects[j]):
                    rects[i] = rects[i] | rects[j]
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return rects


def find_visual_regions(page, blocks):
    """Regions to show as images: embedded pictures, vector charts/diagrams, and tables.

    Returns (rects, absorbed) where absorbed are indices of text blocks that belong to a
    region (chart labels, table cells) and so must not be read as prose.
    """
    import pymupdf
    page_area = page.rect.get_area()
    candidates = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & page.rect
        if r.width >= 40 and r.height >= 40 and r.get_area() < 0.7 * page_area:  # full-page = scan
            candidates.append(r)
    try:
        for r in page.cluster_drawings():  # vector charts, diagrams, ruled tables
            r = r & page.rect
            if r.width >= 60 and r.height >= 40 and r.get_area() < 0.7 * page_area:
                candidates.append(r)
    except Exception:
        pass
    regions = _merge_rects(candidates)

    # Grow regions over small drawings that touch them (frames, axis lines, borders under a picture)
    try:
        small = [d["rect"] & page.rect for d in page.get_drawings()]
    except Exception:
        small = []
    for k, r in enumerate(regions):
        for d in small:
            if d.get_area() < 0.7 * page_area and (r + (-6, -6, 6, 6)).intersects(d):
                r |= d
        regions[k] = r

    # Pull in labels/legends/cells that sit inside a region
    absorbed, kept = set(), []
    for r in regions:
        inside = [i for i, b in enumerate(blocks)
                  if not _CAPTION_RE.match(b[4].strip())
                  and (r & pymupdf.Rect(b[:4])).get_area() >= 0.6 * pymupdf.Rect(b[:4]).get_area()]
        if any(_is_prose_block(blocks[i][4]) for i in inside):
            continue  # a box drawn around a paragraph, not a figure
        for i in inside:
            r |= pymupdf.Rect(blocks[i][:4])
        absorbed.update(inside)
        kept.append(r)

    # Tables without ruling lines: the table-like blocks right below (or above) a "TABLE n" caption
    by_y = sorted(range(len(blocks)), key=lambda i: blocks[i][1])
    for ci in by_y:
        cap = blocks[ci]
        if ci in absorbed or not _TABLE_CAPTION_RE.match(cap[4].strip()):
            continue
        same_col = [i for i in by_y if i != ci and i not in absorbed
                    and min(blocks[i][2], cap[2]) - max(blocks[i][0], cap[0]) > 0]
        for direction in (1, -1):
            seq = [i for i in same_col if (blocks[i][1] > cap[1]) == (direction == 1)]
            seq = seq if direction == 1 else seq[::-1]
            body, last_edge = [], cap[3] if direction == 1 else cap[1]
            for i in seq:
                b = blocks[i]
                gap = b[1] - last_edge if direction == 1 else last_edge - b[3]
                if (gap > 40 or not _is_table_block(b[4]) or _is_prose_block(b[4])
                        or _CAPTION_RE.match(b[4].strip())):
                    break
                body.append(i)
                last_edge = b[3] if direction == 1 else b[1]
            if body:
                r = pymupdf.Rect(blocks[body[0]][:4])
                for i in body:
                    r |= pymupdf.Rect(blocks[i][:4])
                absorbed.update(body)
                kept.append(r)
                break
    return _merge_rects(kept, gap=4), absorbed


def snapshot_region(page, rect) -> str:
    """PNG data URI of a page region (sharp enough for chart text, small enough to embed)."""
    import base64
    clip = (rect + (-8, -8, 8, 8)) & page.rect  # pictures often draw slightly past their reported box
    dpi = 130 if clip.width * 130 / 72 <= 1400 else int(1400 * 72 / clip.width)
    png = page.get_pixmap(clip=clip, dpi=dpi).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def page_text_in_reading_order(page, assets=None, page_index=0) -> str:
    """Text blocks in reading order; on two-column pages, left column before right column.

    When `assets` is given, images/charts/tables are captured into it and replaced in the
    text by [[SR-ASSET:id]] marker paragraphs at their reading-order position.
    """
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if assets is not None:
        regions, absorbed = find_visual_regions(page, blocks)
        blocks = [b for i, b in enumerate(blocks) if i not in absorbed]
        for k, r in enumerate(regions):
            asset_id = f"p{page_index + 1}-{k + 1}"
            assets[asset_id] = snapshot_region(page, r)
            blocks.append((r.x0, r.y0, r.x1, r.y1, f"[[SR-ASSET:{asset_id}]]", -1, 0))
    width = page.rect.width
    mid, tol = width / 2, width * 0.02
    left = [b for b in blocks if b[2] <= mid + tol]
    right = [b for b in blocks if b[0] >= mid - tol]
    column_chars = sum(len(b[4]) for b in left + right)
    two_columns = len(left) >= 2 and len(right) >= 2 and column_chars > 0.5 * sum(len(b[4]) for b in blocks)
    if not two_columns:
        ordered = sorted(blocks, key=lambda b: (round(b[1]), b[0]))
    else:
        # Full-width blocks (title, abstract, wide figures) split the page into bands;
        # within each band read the whole left column, then the whole right column.
        full = sorted((b for b in blocks if b not in left and b not in right), key=lambda b: b[1])
        pending_l = sorted(left, key=lambda b: b[1])
        pending_r = sorted(right, key=lambda b: b[1])
        ordered = []
        for f in full:
            ordered += [b for b in pending_l if b[1] < f[1]] + [b for b in pending_r if b[1] < f[1]]
            pending_l = [b for b in pending_l if b[1] >= f[1]]
            pending_r = [b for b in pending_r if b[1] >= f[1]]
            ordered.append(f)
        ordered += pending_l + pending_r
    return "\n\n".join(b[4].strip() for b in ordered)


def extract_pdf_pages(pdf_bytes, page_indices, cache, ocr_pages, assets=None):
    """Extract the given pages into cache {index: text}; OCR pages with no text layer.
    Figures, charts and tables are captured into `assets` (see page_text_in_reading_order)."""
    import pymupdf
    todo = [p for p in page_indices if p not in cache]
    if not todo:
        return
    bar = st.progress(0.0, text=f"Extracting text… 0 of {len(todo)} pages")
    ocr_available = True
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for k, p in enumerate(todo, 1):
            page = doc[p]
            page_assets = {}
            page_text = page_text_in_reading_order(page, page_assets, p)
            real_text = ASSET_MARKER_RE.sub("", page_text)
            # Little or no text layer → the page is an image of text (a scan, or text drawn as shapes
            # by "Print to PDF"); OCR it. Its "figures" are really the text, so they're dropped —
            # unless OCR finds almost nothing, i.e. the page genuinely is just a picture or chart.
            if len(real_text.strip()) < 20 and ocr_available:
                bar.progress((k - 1) / len(todo), text=f"Reading page {p + 1} with OCR (image-only page)… "
                                                        f"{k} of {len(todo)}")
                try:
                    ocr_text = ocr_page(page)
                    if len(ocr_text.strip()) >= 50 or not page_assets:
                        page_text, page_assets = ocr_text, {}
                        ocr_pages.add(p)
                except ImportError:
                    ocr_available = False
                    st.warning("Some pages are images of text, but OCR isn't installed "
                               "(pip install rapidocr onnxruntime). Those pages were skipped.")
            if assets is not None:
                assets.update(page_assets)
            cache[p] = page_text
            bar.progress(k / len(todo), text=f"Extracting text… {k} of {len(todo)} pages")
    bar.empty()


def _edge_key(line: str) -> str:
    return re.sub(r'\d+', '#', line.strip().lower())


def clean_pdf_pages(pages: list[str]) -> list[str]:
    """Remove running headers/footers (lines repeated at page edges) and bare page numbers."""
    def edge_lines(text):
        idx = [i for i, l in enumerate(text.split("\n")) if l.strip()]
        return set(idx[:2] + idx[-2:])

    counts = {}
    for text in pages:
        lines = text.split("\n")
        for key in {_edge_key(lines[i]) for i in edge_lines(text)}:
            counts[key] = counts.get(key, 0) + 1
    repeated = {k for k, c in counts.items() if len(pages) >= 3 and c >= 0.5 * len(pages) and len(k) > 1}

    cleaned = []
    for text in pages:
        lines = text.split("\n")
        drop = {i for i in edge_lines(text)
                if not ASSET_MARKER_RE.fullmatch(lines[i].strip())  # never drop a figure
                and _edge_key(lines[i]) in repeated
                or re.fullmatch(r'[-–—\s]*(?:\d{1,4}|[ivxlc]{1,6})[-–—\s]*', lines[i].strip(), re.I)}
        cleaned.append("\n".join(l for i, l in enumerate(lines) if i not in drop))
    return cleaned

# ─────────────────────────────────────────────
# 10. STREAMLIT APP
# ─────────────────────────────────────────────

def show_html(page_html: str):
    # Images/charts/tables captured from the current PDF replace their [[SR-ASSET:id]] markers
    page_html = inject_assets(page_html, (st.session_state.get("pdf") or {}).get("assets", {}))
    if hasattr(st, "iframe"):  # st.components.v1.html is deprecated in newer Streamlit
        st.iframe(page_html, height=800)
    else:
        st.components.v1.html(page_html, height=800, scrolling=True)


def _sync_paste():
    # Mirror the text box exactly — clearing it clears the loaded text too
    st.session_state.loaded_text = st.session_state.paste_area


def show_results(res: dict, conf_threshold: float):
    for msg in res["warnings"]:
        st.warning(msg)
    st.divider()
    if res["fallback"]:
        st.info("This text doesn't have strong hierarchical structure — showing original layout with spacing.")
        show_html(res["html_orig"])
    else:
        # ── Tabbed display ── (rendered here so the confidence slider applies without re-analyzing)
        tab_struct, tab_orig = st.tabs(["📐 StructRead", "📄 Original"])
        with tab_struct:
            show_html(render_structread(res["sentences"], res["labels"], conf_threshold, res["inserts"]))
        with tab_orig:
            show_html(res["html_orig"])

    # ── Label inspection ──
    labels, sentences = res["labels"], res["sentences"]
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


def main():
    st.set_page_config(
        page_title="StructRead",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── API key (server-side only) ──
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets.toml present
        secret_key = ""
    api_key = os.environ.get("GROQ_API_KEY") or secret_key
    if not api_key:
        st.error("GROQ_API_KEY not set. Add it to your environment variables or Streamlit secrets.")
        st.stop()

    # ── Session state init ──
    if "loaded_text" not in st.session_state:
        st.session_state.loaded_text = ""

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## StructRead")
        st.caption("Structural offloading for reading persistence")
        st.divider()

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
        st.text_area(
            "Paste your text here", height=250,
            placeholder="Paste a paragraph, section, or chapter…",
            key="paste_area",
            on_change=_sync_paste,
        )

    elif input_method == "Upload PDF":
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")
        if uploaded:
            pdf = st.session_state.get("pdf")
            if not pdf or pdf["id"] != uploaded.file_id:
                # New file: remember it; pages are extracted lazily and cached
                pdf_bytes = uploaded.getvalue()
                try:
                    import pymupdf
                    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                        page_count = doc.page_count
                except Exception as e:
                    st.error(f"Couldn't open this PDF: {e}")
                    st.stop()
                pdf = {"id": uploaded.file_id, "bytes": pdf_bytes, "pages": page_count,
                       "cache": {}, "ocr": set(), "assets": {}}
                st.session_state.pdf = pdf

            total = pdf["pages"]
            first, last = 1, total
            if total > 1:
                first, last = st.slider(
                    "Pages to analyze", 1, total, (1, min(total, 30)), key=f"range_{pdf['id']}",
                    help="Long documents take a while (about a minute per chunk) — start with a section.")
                if total > 30:
                    st.caption(f"This PDF has {total} pages; starting with pages 1–30. "
                               f"Widen the range to include more.")
            selected = list(range(first - 1, last))
            extract_pdf_pages(pdf["bytes"], selected, pdf["cache"], pdf["ocr"], pdf["assets"])
            pages = clean_pdf_pages([pdf["cache"][p] for p in selected])
            pdf_text = "\n\n".join(p for p in pages if p.strip())
            ocr_count = sum(1 for p in selected if p in pdf["ocr"])
            if ocr_count:
                st.caption(f"{ocr_count} of {len(selected)} pages were read with OCR — "
                           f"expect occasional recognition errors.")
            st.session_state.loaded_text = pdf_text
            if pdf_text:
                n_visuals = len(ASSET_MARKER_RE.findall(pdf_text))
                plain = ASSET_MARKER_RE.sub("[figure / chart / table]", pdf_text)
                visuals_note = f" plus {n_visuals} figures, charts and tables" if n_visuals else ""
                st.success(f"Extracted {len(plain)} characters{visuals_note} from pages {first}–{last}.")
                with st.expander("Preview extracted text"):
                    st.text(plain[:2000] + ("…" if len(plain) > 2000 else ""))
            else:
                st.error(
                    "No text could be extracted from this PDF. It's probably a scan or was saved with "
                    "\"Microsoft Print to PDF\", which stores pages as images/shapes instead of text. "
                    "Try the original PDF or EPUB export, or copy the text and use \"Paste text\"."
                )

    elif input_method == "Use sample":
        st.markdown("A paragraph about sleep deprivation and cognitive performance.")
        if st.button("Load sample text"):
            st.session_state.loaded_text = SAMPLE_TEXT
            st.rerun()

    # ── Show status and always show button ──
    text = st.session_state.loaded_text

    if text:
        st.success(f"✓ Text loaded — {len(text)} characters, ~{estimate_tokens(text)} tokens")
        # Up-front cost estimate, before anything is sent to Groq
        pre_sentences, _, _ = prepare_text(text)
        pre_chunks = len(chunk_sentences(pre_sentences, max_tokens=chunk_size)) if pre_sentences else 0
        if pre_chunks > 1:
            mins = estimate_minutes(pre_chunks)
            msg = (f"Analysis will take about **{mins} min** ({pre_chunks} chunks with a 65s pause "
                   f"between each). A larger chunk size in the sidebar means fewer pauses.")
            if mins > 15:
                st.warning(msg + " For long documents, consider analyzing a smaller page range first.")
            else:
                st.caption(msg)
    else:
        st.info("Paste text, upload a PDF, or load the sample to get started.")

    st.divider()

    # Button is ALWAYS visible
    if st.button("🔍 Analyze structure", type="primary", use_container_width=True):
        st.session_state.pop("result", None)
        if not text:
            st.error("No text loaded. Paste, upload, or load the sample first.")
            return

        # Step 1: Split sentences — headings, captions/tables and list-like sections
        # (references, contents, index…) are kept as-is and never sent to the LLM
        with st.spinner("Splitting sentences…"):
            sentences, inserts, kept_titles = prepare_text(text)
        kept_note = f" · shown as-is: {', '.join(kept_titles)}" if kept_titles else ""
        st.caption(f"{len(sentences)} sentences identified{kept_note}")

        if not sentences:
            st.info("Nothing to restructure — this text is only list-like sections. Showing it as-is.")
            show_html(render_paragraphs(text))
            return

        # Step 2: Chunk
        chunks = chunk_sentences(sentences, max_tokens=chunk_size)
        total_chunks = len(chunks)

        if total_chunks > 1:
            st.info(
                f"📦 Text split into **{total_chunks} chunks** for rate limits. "
                f"Estimated time: **~{estimate_minutes(total_chunks)} min** "
                f"(includes a 65s pause between chunks). "
                f"A larger chunk size in the sidebar means fewer pauses."
            )

        # Step 3: Process
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            all_labels, failed, stop_message = process_all_chunks(
                sentences, chunks, api_key, model, progress_bar, status_text
            )
        except Exception as e:
            st.error(f"Groq API error: {e}")
            return

        if not all_labels:
            st.error(stop_message or "Groq didn't return any usable labels. Try again, or try another model.")
            return

        # Step 4: Render
        labels = validate_labels(all_labels, len(sentences))

        warnings = []
        if stop_message:
            warnings.append(stop_message)
        for first_id, last_id, reason in failed:
            if reason != "rate limit reached":
                warnings.append(f"Sentences {first_id}–{last_id} couldn't be analyzed ({reason}) "
                                f"and are shown as plain text.")

        # Low-confidence share, ignoring sentences that were never analyzed
        failed_ids = {i for a, b, _ in failed for i in range(a, b + 1)}
        analyzed = [lb for lb in labels if lb["id"] not in failed_ids]
        low_conf = sum(1 for lb in analyzed if lb["confidence"] < 0.6)
        fallback = bool(analyzed) and low_conf / len(analyzed) > 0.6

        st.session_state.result = {
            "text": text,
            "fallback": fallback,
            "inserts": inserts,
            "html_orig": render_paragraphs(text),  # full text, including the as-is sections
            "labels": labels,
            "sentences": sentences,
            "warnings": warnings,
        }

    # Results persist across reruns (sliders, tabs…) until the text changes or Analyze runs again
    result = st.session_state.get("result")
    if result and result["text"] == text:
        show_results(result, conf_threshold)


if __name__ == "__main__":
    main()
