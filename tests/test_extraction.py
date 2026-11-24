"""Extraction regression tests (no LLM): each test_data/<name>.expected.json describes what the
text extracted from a PDF must look like. A PDF that isn't present (PDFs are gitignored) is skipped.

Entries listed under "known_failures" are xfail(strict=True): the test says so when a fix makes
one of them pass, and then it should be removed from the JSON.
"""
import glob
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import app  # noqa: E402

SPECS = []
for _path in sorted(glob.glob(os.path.join(ROOT, "test_data", "*.expected.json"))):
    with open(_path, encoding="utf-8") as _f:
        SPECS.append(json.load(_f))

_EXTRACTED = {}


def extracted_text(spec) -> str:
    """The app's final PDF text for every page of the spec's PDF (cached across tests)."""
    name = spec["pdf"]
    if name not in _EXTRACTED:
        path = next((p for p in (os.path.join(ROOT, name), os.path.join(ROOT, "test_data", name))
                     if os.path.exists(p)), None)
        if path is None:
            pytest.skip(f"{name} is not present")
        import pymupdf
        with open(path, "rb") as f:
            data = f.read()
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            pages = list(range(doc.page_count))
        cache, ocr, assets = {}, set(), {}
        app.extract_pdf_pages(data, pages, cache, ocr, assets)
        _EXTRACTED[name], _ = app.assemble_pdf_text([cache[p] for p in pages], True)
    return _EXTRACTED[name]


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def flat(text: str) -> str:
    """Single line breaks (inside a paragraph) become spaces; paragraph breaks stay."""
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def cases(key, known_key="known_failures"):
    """pytest params (spec, item), xfail(strict) for items the spec lists as known failures."""
    out = []
    for spec in SPECS:
        for item in spec.get(key, []):
            marks = [pytest.mark.xfail(strict=True, reason="known failure")] \
                if item in spec.get(known_key, []) else []
            out.append(pytest.param(spec, item, marks=marks, id=f"{spec['pdf']}::{item[:50]}"))
    return out


def specs():
    return [pytest.param(s, id=s["pdf"]) for s in SPECS]


def index_of(paras, start):
    return next((i for i, p in enumerate(paras) if flat(p).startswith(start)), None)


@pytest.mark.parametrize("spec", specs())
def test_title_is_marked(spec):
    assert f"[[SR-TITLE]] {spec['title']}" in flat(extracted_text(spec))


@pytest.mark.parametrize("spec", specs())
def test_author_line_is_meta_and_follows_the_title(spec):
    paras = paragraphs(extracted_text(spec))
    t = index_of(paras, "[[SR-TITLE]]")
    m = index_of(paras, "[[SR-META]] " + spec["meta_start"])
    assert t is not None and m is not None and t < m
    assert not any(flat(p).startswith("[[SR-META]]") and p[12:13].islower() for p in paras)


@pytest.mark.parametrize("spec", specs())
def test_abstract_is_one_complete_separate_paragraph(spec):
    paras = paragraphs(extracted_text(spec))
    a = index_of(paras, spec["abstract_start"])
    assert a is not None, "abstract start not found as the start of a paragraph"
    assert flat(paras[a]).endswith(spec["abstract_end"]), "abstract is truncated or merged with what follows"
    meta = index_of(paras, "[[SR-META]]")
    intro = index_of(paras, spec["headings"][0])
    assert meta < a < intro
    for p in paras[a:a + 1]:
        assert "[[SR-" not in p and spec["headings"][0] not in p and spec["title"] not in p


@pytest.mark.parametrize("spec", specs())
def test_index_terms_are_a_separate_paragraph(spec):
    paras = paragraphs(extracted_text(spec))
    i = index_of(paras, spec["index_terms_start"])
    assert i is not None
    a = index_of(paras, spec["abstract_start"])
    assert a < i < index_of(paras, spec["headings"][0])
    assert flat(paras[i]).endswith(".") and "INTRODUCTION" not in paras[i]


@pytest.mark.parametrize("spec,heading", cases("headings"))
def test_heading_is_its_own_paragraph(spec, heading):
    assert heading in [flat(p) for p in paragraphs(extracted_text(spec))]


@pytest.mark.parametrize("spec", specs())
def test_headings_come_in_order(spec):
    flats = [flat(p) for p in paragraphs(extracted_text(spec))]
    positions = []
    for h in spec["headings"]:
        if h in spec.get("known_failures", []):
            continue
        positions.append(flats.index(h))
    assert positions == sorted(positions)


@pytest.mark.parametrize("spec,sentence", cases("must_contain"))
def test_sentence_is_unbroken(spec, sentence):
    assert sentence in flat(extracted_text(spec))


@pytest.mark.parametrize("spec,bad", cases("must_not_contain"))
def test_text_has_no_leaked_markup_or_glued_words(spec, bad):
    assert bad not in extracted_text(spec)
