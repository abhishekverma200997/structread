"""Tests for StructRead's pure logic (no Groq calls, no embedding model).

Run from the repo root:  pip install pytest && pytest
"""
import os
import random
import sys
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────

class _ReadingText(HTMLParser):
    """Text a reader sees in the document body: skips <style>/<script>, the legend, the collapsed-section
    previews (which deliberately repeat topic sentences), section titles and buttons."""
    SKIP_TAGS = {"style", "script", "button"}
    SKIP_CLASSES = {"sr-key", "sr-preview", "sr-title", "sr-check", "sr-progress"}
    VOID = {"meta", "img", "br", "hr", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack, self.out = [], []

    def handle_starttag(self, tag, attrs):
        if tag in self.VOID:
            return
        classes = set((dict(attrs).get("class") or "").split())
        self.stack.append(tag in self.SKIP_TAGS or bool(classes & self.SKIP_CLASSES))

    def handle_endtag(self, tag):
        if tag not in self.VOID and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        if not any(self.stack):
            self.out.append(data)


def reading_text(page_html: str) -> str:
    parser = _ReadingText()
    parser.feed(page_html)
    return " ".join(" ".join(parser.out).split())


def label(i, role="subordinate", indent=0, group=None, parent_id=None, confidence=0.9, **extra):
    return {"id": i, "role": role, "indent": indent, "group": group, "parent_id": parent_id,
            "confidence": confidence, "transition_type": None, "relation_to_parent": None, **extra}


def assert_every_sentence_once_in_order(page_html, sentences):
    text = reading_text(page_html)
    pointer = 0
    for s in sentences:
        assert text.count(s) == 1, f"sentence shown {text.count(s)} times: {s!r}"
        at = text.find(s, pointer)
        assert at >= 0, f"sentence out of order: {s!r}"
        pointer = at + len(s)


# ── sentence splitting ───────────────────────────────────────────────────

def test_split_keeps_abbreviations_and_initials_together():
    text = "Dr. Smith et al. measured it in the U.S. last year. J. K. Rowling disagreed. It rose by 3.5 percent."
    assert app.split_sentences(text) == [
        "Dr. Smith et al. measured it in the U.S. last year.",
        "J. K. Rowling disagreed.",
        "It rose by 3.5 percent.",
    ]


def test_split_loses_no_characters():
    text = app.SAMPLE_TEXT
    assert "".join(app.split_sentences(text)).replace(" ", "") == text.replace(" ", "")


# ── chunking ─────────────────────────────────────────────────────────────

def test_chunks_cover_every_sentence_once_and_respect_the_cap():
    sentences = [f"Sentence number {i} is here." for i in range(205)]
    chunks = app.chunk_sentences(sentences, max_tokens=app.CHUNK_TOKENS)
    assert [i for c in chunks for i in c] == list(range(205))
    assert all(len(c) <= app.MAX_SENTENCES_PER_CHUNK for c in chunks)


# ── label parsing / validation ───────────────────────────────────────────

def test_parse_labels_salvages_truncated_json():
    raw = '{"labels": [{"id": 1, "role": "topic"}, {"id": 2, "role": "subordinate"}, {"id": 3, "ro'
    assert [lb["id"] for lb in app.parse_labels(raw)] == [1, 2]


def test_validate_fills_gaps_and_normalizes():
    raw = [{"id": "1", "role": "TOPIC", "indent": 0, "confidence": "92%"},
           {"id": 3, "role": "nonsense", "indent": 7, "parent_id": 3, "confidence": 2}]
    out = app.validate_labels(raw, 3)
    assert [lb["id"] for lb in out] == [1, 2, 3]
    assert out[0]["role"] == "topic" and out[0]["confidence"] == 0.92
    assert out[1]["role"] == "subordinate" and out[1]["confidence"] == 0.5   # missing → safe default
    assert out[2]["role"] == "subordinate" and out[2]["parent_id"] is None   # self-parent removed


def test_indent_never_exceeds_one_level():
    out = app.validate_labels([{"id": 1, "role": "subordinate", "indent": 5, "confidence": 0.9}], 1)
    assert out[0]["indent"] == app.MAX_INDENT == 1


def test_vote_labels_takes_the_majority_role():
    samples = [[{"id": 1, "role": "topic"}], [{"id": 1, "role": "topic"}], [{"id": 1, "role": "deferrable"}]]
    voted = app.vote_labels(samples)
    assert voted[0]["role"] == "topic" and voted[0]["confidence"] == 0.67


def test_check_label_ids_drops_foreign_and_duplicate_ids():
    labels = [{"id": 5, "role": "topic"}, {"id": 5, "role": "subordinate"}, {"id": 99, "role": "topic"}]
    kept = app.check_label_ids(labels, {5, 6}, "chunk 1")
    assert kept == [{"id": 5, "role": "topic"}]


# ── reconcile + bold budget ──────────────────────────────────────────────

def test_reconcile_makes_the_opening_sentence_a_topic():
    labels = [label(1), label(2)]
    out = app.reconcile_labels(labels, {}, [[0, 1]], None, None, [])
    assert out[0]["role"] == "topic"


def test_bold_budget_demotes_the_least_central_topics_but_not_the_first():
    n = 20
    labels = [label(i + 1, role="topic") for i in range(n)]
    centrality = [0.0] + [i / n for i in range(1, n)]  # sentence 1 is the least central
    out = app.enforce_topic_budget(labels, centrality, max_share=0.2, conf_threshold=0.6)
    topics = [lb["id"] for lb in out if lb["role"] == "topic"]
    assert len(topics) == 4 and 1 in topics           # 20% of 20, opening sentence kept
    assert topics[1:] == [18, 19, 20]                  # the most central ones survive
    assert app.topic_share(out) == 0.2
    assert all(lb["indent"] == 0 for lb in out if lb["role"] == "subordinate")  # demoted → plain


def test_bold_budget_leaves_short_or_sparse_texts_alone():
    short = [label(i + 1, role="topic") for i in range(5)]
    assert all(lb["role"] == "topic" for lb in app.enforce_topic_budget(short))
    sparse = [label(i + 1, role="topic" if i % 10 == 0 else "subordinate", indent=1) for i in range(30)]
    assert sum(lb["role"] == "topic" for lb in app.enforce_topic_budget(sparse)) == 3


# ── rendering: the "same words, new layout" guarantee ────────────────────

def test_sample_like_layout_keeps_every_sentence_once_in_order():
    sentences = [f"Sentence number {i} is here." for i in range(1, 9)]
    labels = [label(1, "topic"), label(2, "coordinate", group="A"), label(3, "subordinate", 1, parent_id=2),
              label(4, "coordinate", group="A"), label(5, "deferrable", 1, parent_id=4),
              label(6, "transition", transition_type="comparison"),
              label(7, "subordinate", 1, parent_id=6, relation_to_parent="explains why"),
              label(8, "topic", confidence=0.3)]
    page = app.render_structread(sentences, labels, app.CONF_THRESHOLD)
    assert_every_sentence_once_in_order(page, sentences)
    assert "<ol>" in page and 'class="s-topic"' in page and 'class="s-deferrable"' in page


def test_nothing_beyond_the_text_is_drawn_for_relations_or_transition_types():
    sentences = ["However, this is a summary.", "It explains the claim."]
    labels = [label(1, "transition", transition_type="comparison"),
              label(2, "subordinate", 1, parent_id=1, relation_to_parent="explains why")]
    page = app.render_structread(sentences, labels, app.CONF_THRESHOLD)
    body = reading_text(page)
    assert "explains why" not in body and "→" not in body
    assert "tt-comparison" not in page and "relation-hint" not in page and "indent-2" not in page


def test_subordinate_with_indent_zero_renders_plain():
    page = app.render_structread(["One.", "Two."], [label(1, "topic"), label(2, "subordinate", 0)])
    assert '<div class="s-plain">Two.</div>' in page


def test_source_text_is_escaped_not_trusted_as_html():
    page = app.render_structread(["<script>alert(1)</script> & more."], [label(1, "topic")])
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page


def test_random_labels_never_drop_duplicate_or_reorder_a_sentence():
    rng = random.Random(7)
    roles = sorted(app._ROLES)
    for _ in range(300):
        n = rng.randint(1, 60)
        sentences = [f"Sentence number {i} is here." for i in range(1, n + 1)]
        raw = [{"id": i, "role": rng.choice(roles), "indent": rng.randint(0, 3),
                "group": rng.choice(["A", "B", None]), "parent_id": rng.choice([None, rng.randint(1, n)]),
                "confidence": rng.random(), "transition_type": rng.choice([None, "temporal"]),
                "relation_to_parent": rng.choice([None, "adds detail"])}
               for i in range(1, n + 1) if rng.random() > 0.05]            # ~5% of labels missing
        labels = app.enforce_topic_budget(app.validate_labels(raw, n))
        inserts = {}
        if n > 12 and rng.random() < 0.5:                                    # headings → collapsible sections
            for k in sorted(rng.sample(range(n), 3)):
                title = f"{k + 1} Heading Title"
                inserts[k] = [{"kind": "heading", "title": title, "html": app.render_heading(title), "level": 1}]
        page = app.render_structread(sentences, labels, app.CONF_THRESHOLD, inserts)
        assert_every_sentence_once_in_order(page, sentences)


# ── segmentation ─────────────────────────────────────────────────────────

def test_references_and_headings_are_not_sent_to_the_llm():
    text = ("1 Introduction\n\nSleep matters for memory. It also matters for mood.\n\n"
            "References\n[1] Smith, J. (2020). A paper. Journal.\n[2] Doe, A. (2021). Another paper.")
    sentences, inserts, kept = app.prepare_text(text)
    assert sentences == ["Sleep matters for memory.", "It also matters for mood."]
    assert kept == ["References"]
    assert any(isinstance(b, dict) and b["kind"] == "heading" for blocks in inserts.values() for b in blocks)


# ── join_split_paragraphs / check_extraction ─────────────────────────────

def join(*paras):
    return app.join_split_paragraphs("\n\n".join(paras)).split("\n\n")


def test_join_rejoins_a_paragraph_split_mid_sentence():
    out = join("Signals must cover various environments [17] and distance ranges",
               "with balanced representation, requiring extensive measurement campaigns.")
    assert out == ["Signals must cover various environments [17] and distance ranges "
                   "with balanced representation, requiring extensive measurement campaigns."]


def test_join_moves_a_figure_or_footnote_marker_after_the_joined_paragraph():
    out = join("Signals must cover distance ranges", "[[SR-ASSET:p1-1]]", "[[SR-FOOTNOTE]] A note.",
               "with balanced representation.", "Next paragraph.")
    assert out == ["Signals must cover distance ranges with balanced representation.",
                   "[[SR-ASSET:p1-1]]", "[[SR-FOOTNOTE]] A note.", "Next paragraph."]


def test_join_keeps_a_trailing_hyphen_without_a_space():
    assert join("a model for compre-", "hensive training works") == ["a model for compre-hensive training works"]


def test_join_leaves_headings_captions_references_and_uppercase_alone():
    assert len(join("I. INTRODUCTION", "the first paragraph")) == 2
    assert len(join("Fig. 1. Signal propagation in different", "environments are shown")) == 2
    assert len(join("some text without an end", "[1] Smith, J. A paper. Journal, 2020.")) == 2
    assert len(join("some text without an end", "Another paragraph starts here.")) == 2
    assert len(join("some text without an end", "17 more things happen here.")) == 2
    assert len(join("[[SR-META]] Jane Doe, Somewhere", "university of lowercase")) == 2
    assert len(join("some text without an end", "[[SR-META]] x")) == 2


def test_join_does_not_join_after_terminal_punctuation_or_quotes():
    for end in ["It ends here.", "Really?", "Stop!", "As follows:", 'He said "stop."', "She said “stop.”"]:
        assert len(join(end, "and then more.")) == 2, end


def test_join_chains_across_several_splits_and_never_changes_a_word():
    text = "\n\n".join(["one two three four", "five six seven eight", "nine ten."])
    out = app.join_split_paragraphs(text)
    assert out == "one two three four five six seven eight nine ten."


def test_join_prints_each_join_when_debugging(monkeypatch, capsys):
    monkeypatch.setenv("STRUCTREAD_DEBUG_JOINS", "1")
    join("The long sentence about distance ranges", "with balanced representation.")
    assert "JOIN: …The long sentence about distance ranges | with balanced representation.…" in capsys.readouterr().out
    monkeypatch.delenv("STRUCTREAD_DEBUG_JOINS")
    join("The long sentence about distance ranges", "with balanced representation.")
    assert capsys.readouterr().out == ""


def test_check_extraction_is_quiet_on_clean_text():
    text = "[[SR-TITLE]] A Title\n\nA clean paragraph.\n\n[[SR-ASSET:p1-1]]\n\nAnother clean one."
    assert app.check_extraction(text, text) == []


def test_check_extraction_flags_missing_title_glue_markup_lowercase_and_changed_words():
    raw = "Some text laborI. INTRODUCTION ~~x~~ <!-- c --> <br>"
    final = raw + "\n\nthis paragraph has a lowercase start"
    warnings = " | ".join(app.check_extraction(raw, final))
    for expected in ["no title", "glued words", "laborI", "~~", "<!--", "<br", "start lowercase", "words changed"]:
        assert expected in warnings, expected
    assert "no title" not in " | ".join(app.check_extraction(raw, raw, first_page_included=False))


def test_check_extraction_ignores_markers_and_whitespace_when_comparing_words():
    raw = "[[SR-TITLE]] A Title\n\nfirst half\n\n[[SR-ASSET:p1-1]]\n\nsecond half."
    warnings = app.check_extraction(raw, app.join_split_paragraphs(raw))
    assert not any("words changed" in w for w in warnings)


def test_join_places_a_caption_and_a_stray_page_number_after_the_joined_paragraph():
    out = join("In this section we present the hierarchical LVM that", "2", "[[SR-ASSET:p2-1]]",
               "Fig. 2. Graphical model for the signal.", "enables to generate realistic signals.")
    assert out == ["In this section we present the hierarchical LVM that enables to generate realistic signals.",
                   "2", "[[SR-ASSET:p2-1]]", "Fig. 2. Graphical model for the signal."]


def test_join_skips_math_and_algorithm_lines_but_not_prose_or_bibliography():
    prose = "We maximise the evidence lower bound over the model parameters and then"
    assert len(join(prose, "p(zd, ze|d, k) = p(zd|d)p(ze|k) (2)")) == 2            # equation after prose
    assert len(join("the bound holds when ϕ = {ϕd, ϕe} are fixed and the", "decoder parameters θ∗ stay fixed")) == 2
    assert len(join('Smith, J., and Doe, A., “A review of', 'indoor localization techniques,” IEEE, 2020.')) == 1
    assert len(join("Signals must cover various environments and distance ranges", "with balanced representation.")) == 1


def test_join_needs_a_real_first_paragraph_not_an_orphan_fragment():
    assert len(join("+ DKL", "log p(x) holds")) == 2


def test_check_extraction_does_not_count_math_lines_as_lowercase_starts():
    text = "\n\n".join(["[[SR-TITLE]] T", "A normal paragraph.", "ϕ∗, θ∗, φ∗= arg max", "p(zd) = N(zd; 0, ϵdI) (9)"])
    assert not any("start lowercase" in w for w in app.check_extraction(text, text))


def test_the_ui_hides_the_glued_word_warning_below_ten_words():
    few = ["3 possibly glued words, e.g. ResNet", "2 paragraphs start lowercase, e.g. 'a'"]
    many = ["12 possibly glued words, e.g. a, b"]
    assert app.warnings_for_ui(few) == ["2 paragraphs start lowercase, e.g. 'a'"]
    assert app.warnings_for_ui(many) == many


def test_check_extraction_does_not_count_fragments_under_four_words_as_lowercase_starts():
    text = "\n\n".join(["[[SR-TITLE]] T", "A normal paragraph.", "d )I", "log p(x|zd)"])
    assert not any("start lowercase" in w for w in app.check_extraction(text, text))
    text += "\n\nthis one is a real lowercase start"
    assert any("1 paragraphs start lowercase" in w for w in app.check_extraction(text, text))
