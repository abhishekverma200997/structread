# StructRead — Project State

_Last updated: 2026-11-24_

## Overview
StructRead is an ADHD reading assistant. An LLM labels the discourse role of every sentence, then the app re-renders **the same words in a new layout** that makes the structure visible. It never rephrases, adds or removes text.

- **Stack:** Streamlit (`app.py`), Groq LLM, sentence-transformers (centrality), PyMuPDF (PDF extraction), RapidOCR (image-only pages)
- **Roles:** topic (bold), coordinate, subordinate (indented, max 1 level), deferrable (muted), transition
- **Pipeline:** PDF → `extract_pdf_pages` (`page_text_in_reading_order`, `find_visual_regions`, `mark_front_matter`) → `clean_pdf_pages` → `prepare_text` → `split_sentences` → chunking → centrality → global pass → local labeling → validation → reconciliation → HTML render
- **Internal markers:** `[[SR-TITLE]]`, `[[SR-META]]` (authors), `[[SR-FOOTNOTE]]`, `[[SR-ASSET:id]]` (figure/table snapshots)
- **Deployment:** local only (Windows). No cloud deployment planned for now.

## Running it (Windows PowerShell)
- Local repo: `C:\Users\vaabh\OneDrive\Desktop\ADHD\structread`
- Normal: `streamlit run app.py` → http://localhost:8501
- Dev mode (preprocessed test PDF, no upload): `$env:STRUCTREAD_DEV="1"; streamlit run app.py`
- Leave dev mode: `$env:STRUCTREAD_DEV=""`
- Tests: `python -m pytest tests -q` (extraction tests need `Paper1.pdf` in the project folder; skipped if missing)
- Debug paragraph joins: `$env:STRUCTREAD_DEBUG_JOINS="1"`
- Long PDFs are slow: Groq rate limit adds a ~65 s pause per chunk (~19 min for a 6-page paper).

## Current state
**All changes are uncommitted.** Nothing is committed or pushed until the user approves.

Applied (from `structread-improvements.patch`):
- Prompt rule 9 (connectives), muted text contrast `#6B6A63`, bold budget (`TOPIC_BUDGET = 0.20`), `MAX_INDENT = 1`, relation hints removed, `.s-plain` class, `:focus-visible` styles, reduced-motion support, README rewrite, unit tests

Since then:
- Front-matter fix: a block starting lowercase is never tagged `[[SR-META]]`
- Golden test file `test_data/Paper1.expected.json` (title, 23 headings, abstract start/end, index terms, must_contain, must_not_contain, known_failures)
- `.gitignore`: `!test_data/*.expected.json` so golden files are tracked; PDFs stay ignored

Done since then (still uncommitted):
- `join_split_paragraphs(text)`: applied once to the final PDF text (via `assemble_pdf_text`, both the upload path and the dev-mode path). Joins paragraph A and the next paragraph B only when:
  - A has at least 4 words and does not end in `. ? ! :` or a closing quote after those
  - B starts with a lowercase letter a-z
  - neither is an `[[SR-…]]` marker, a heading (`_is_heading`) or a caption (`_CAPTION_RE`)
  - **math guard:** neither looks like math/algorithm text (`_looks_like_math`: letter ratio < 0.6 `MATH_LETTER_RATIO`, or math symbols/Greek letters ≥ 3 % `MATH_SYMBOL_DENSITY`)
  - Paragraphs **between** A and B that are figures (`[[SR-ASSET:…]]`), footnotes, captions or a stray bare page number ("2") are "floating": A+B join and those paragraphs move to just after the joined paragraph, in original order.
  - A trailing "-" is kept and joins with no space; otherwise one space. Words are never changed.
  - `STRUCTREAD_DEBUG_JOINS=1` prints each join (last 40 chars of A | first 40 chars of B).
- `check_extraction(raw, final, first_page_included)`: warnings for no title, glued words (2+ lowercase then uppercase), leaked `~~` / `<!--` / `<br`, lowercase paragraph starts (ignores math-like paragraphs and paragraphs under 4 words), and any change to the word multiset. The UI shows one caption (`warnings_for_ui`: the glued-words warning only when there are 10 or more); everything prints when `STRUCTREAD_DEV=1`.
- `tests/test_extraction.py`: checks Paper1.pdf against the golden file (title, author line, abstract complete and separate, Index Terms separate, headings in order, must_contain, must_not_contain); `known_failures` entries are `xfail(strict=True)`.

Results on Paper1.pdf (471 paragraphs):
- Joins: 127 without the math guard → **37** with it (11 prose, 26 bibliography). Paragraphs 471 → 434.
- Paragraphs starting lowercase: 138 → 101; 91 of those are math/algorithm lines. `check_extraction` now reports **2** (see backlog).
- The "distance ranges | with balanced representation" pair and the "…LVM | 2 | figure | Fig. 2. caption | enables to generate…" case both join.
- Index Terms stays its own paragraph. No words changed.
- paper1a.pdf (known-bad) warns as intended: no title, 12 glued words.
- **Tests: 70 passed, 1 xfailed** (the wrapped heading).

## Decisions log
| Date | Decision | Why |
|---|---|---|
| 2026-11-24 | Dropped Marker (marker-pdf) | ~4 GB of dependencies (PyTorch, models) for one fix |
| 2026-11-24 | Dropped pymupdf4llm | Added 435 markup artifacts (`~~`, `<!-- -->`, `<br>`) on one paper; fixing its output with regex is fragile |
| 2026-11-24 | Dropped the text-only block-merging heuristic | Without coordinates it merged across columns and broke the title and headings |
| 2026-11-24 | Paragraph join rule: join only when the first part lacks end punctuation and the next starts **lowercase**; never join markers, headings or captions; move figures/footnotes after the joined paragraph; keep hyphens | A wrong merge garbles text; a missed merge is a small glitch. Precision over recall. Never change words. |
| 2026-11-24 | Math guard on joins (letter ratio < 0.6 or math-symbol density ≥ 3 %), tuned on Paper1.pdf | Without it, 123 joins glued equation fragments onto prose. A higher ratio alone (0.7) also dropped valid bibliography joins; symbol density separated them (prose/bibliography 0.000, math 0.06–0.2). Bibliography joins are kept on purpose. |
| 2026-11-24 | Captions, figures, footnotes and stray page numbers float between the halves of a split paragraph; first part needs ≥ 4 words | Fixes the Fig. 2 case; keeps orphan fragments like "+ DKL" from joining |
| 2026-11-24 | `check_extraction` ignores math-like paragraphs and paragraphs under 4 words; UI shows glued-word warnings only at 10+ | Proper names (ResNet, PatchGAN) and equation fragments made the warnings noise; dev mode still prints everything |
| 2026-11-24 | A block starting lowercase is never `[[SR-META]]` | Author/affiliation lines never start lowercase; a paragraph continuation was being tagged as authors |
| 2026-11-24 | Golden-file regression tests on a small fixed PDF set + automatic checks on every PDF | The title/Introduction broke unnoticed because tests used hand-written strings, not real PDFs |
| 2026-11-24 | Local-only deployment | Avoids cloud size limits; fine for a research prototype |

## Known issues / backlog
- **paper1a.pdf (known-bad):** its text layer puts both columns in one block, so lines interleave. Fix later: split blocks that span both columns into lines and sort by x-position.
- **Wrapped headings:** "III. VARIATIONAL INFERENCE METHOD FOR SIGNAL" / "GENERATION" split into two paragraphs. Fix: join consecutive all-caps heading lines. (xfail in golden test)
- **Line-break hyphens:** "exten- sive", "compre- hensive" left as-is (never alter words).
- **Reference false positive:** "S. Salous, Y. Miao, …" detected as a heading by `_is_heading`.
- **Sentence continuing after a display equation:** "required to further estimate the intractable conditional distribution…" is not joined to the prose before the equation, because equation lines sit between them. (The other remaining lowercase start is a bibliography continuation, "modeling methods and tools,” in Inclusive Radio…".)
- **Math detection misses short fragments:** `p(zd, ze)`, `log p(x|zd)`, `d )I` slip past the guard; harmless now (under 4 words, never joined or counted), but a better equation detector would help.
- **Dev-mode cache:** `test_data/processed.json` is keyed by PDF hash and chunk size only, so it does not refresh when extraction logic changes; delete it after extraction changes.
- **paper2a.pdf:** extraction very slow (likely OCR); not yet tested.
- **Project docs:** rewrite Steps 1–4 doc to match the pivot; draft Step 5 evaluation plan (gold set + pilot study).

## Testing
- **Golden set:** `Paper1.pdf` only (`default.pdf` is the same paper). Add a PDF with a different layout (single-column, other publisher) when available.
- A new golden file is drafted by Claude Code and reviewed by the user (title, headings, abstract against the real PDF).
- Every bug found in real use becomes a new golden case.

## Working rules
- **Never commit or push** without the user's explicit approval.
- Test locally on real PDFs before trusting a change; unit tests with hand-written strings aren't enough.
- Plan and decide in Claude chat; implement in Claude Code.
- Keep extraction changes narrow, and check them against the golden tests.
- User preference: brief answers.
