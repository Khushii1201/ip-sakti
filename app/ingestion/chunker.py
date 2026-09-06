"""
Splits statute text at Section/Rule boundaries.

Design note (why this is a line scan, not one big regex):
A single regex with lookahead/lookbehind alternation (tried in earlier
rounds) technically worked but was fragile -- a variable-length lookbehind
plus a greedy character class is exactly the kind of pattern that silently
breaks the next time someone "cleans it up." A heading is fundamentally a
property of a LINE in context (what came before it, what follows it), not
a property of a substring in a flat character stream. Expressing that as an
explicit state machine is clearer, faster to read, and structurally
impossible to break via backtracking -- because there is no backtracking.

The invariant being protected:
  A heading is recognized when ALL of the following hold simultaneously:
    (a) the line contains *only* a Section/Rule reference (plus optional
        trailing punctuation: period, hyphen, or en/em-dash), AND
    (b) either the previous non-empty line is blank (paragraph-start), OR
        the heading is at the very start of the document, OR
        the same line also carries the section title (inline style:
        "Section 3. What are not inventions.--").
  Cross-references that happen to be the only token on a *wrapped* line
  fail condition (b): they are always mid-paragraph (the preceding line
  has real sentence content), so the blank-line test rejects them.

Stress-test repros that each alternative was validated against:
  1. "...as defined under Section 2(1)(l), which..." mid-sentence:
     rejected because the cross-ref is mid-paragraph (no blank before it)
     and has trailing content that is not title-punctuation.
  2. "Section 3\\nWhat are not inventions.--" split across lines:
     accepted as paragraph-heading because blank line precedes it.
  3. A cross-reference that happens to land alone on its own wrapped line:
     rejected because the preceding content line is not blank.
  4. Inline heading "Section 3. What are not inventions.--":
     accepted immediately on the same line via the title-suffix check.

ASSUMPTION TO VERIFY once real corpus text lands:
  Source text uses "Section N" / "Rule N" literally (not bare "3."
  numbering as some gazette PDFs print it).  Normalize \\r\\n to \\n during
  corpus cleanup if the source files come from Windows-authored text.
"""

import re

# Matches a bare Section/Rule reference, e.g. "Section 3" / "Rule 12A".
_REF_RE = re.compile(
    r"^[ \t]*(?P<ref>(?:Section|Rule)\s+\d+[A-Za-z]*)(?P<tail>[^\n]*)$",
    re.IGNORECASE,
)

# Title-punctuation that may follow the ref on the same line for inline style:
# "Section 3. What are not inventions.--"
_TITLE_SUFFIX_RE = re.compile(r"^[ \t]*[.\-\u2014]")


def _classify_line(line: str, prev_blank: bool) -> tuple[str | None, bool]:
    """Return (section_ref_or_None, this_line_is_blank).

    A heading is returned only when the line matches the Section/Rule pattern
    AND the context (prev_blank or inline suffix) confirms it is a real heading
    and not a cross-reference.
    """
    stripped = line.rstrip("\n")
    if not stripped.strip():
        return None, True  # blank line

    m = _REF_RE.match(stripped)
    if m is None:
        return None, False  # ordinary content line

    tail = m.group("tail")

    # Inline heading: ref followed immediately by punctuation + optional title.
    # e.g. "Section 3. What are not inventions.--"
    if tail and _TITLE_SUFFIX_RE.match(tail):
        return m.group("ref").strip(), False

    # Paragraph-start heading: ref is the only token on the line, and we are
    # at the start of a fresh paragraph (blank line before us, or doc start).
    # The title text will arrive on the next line.
    if not tail.strip() and prev_blank:
        return m.group("ref").strip(), False

    # Everything else: cross-reference embedded in prose, or a wrapped line
    # mid-paragraph that coincidentally contains only a ref token.
    return None, False


def chunk_by_section(full_text: str, act_name: str) -> list[dict]:
    """Split statute text at Section/Rule boundaries.

    Deliberately raises instead of silently falling back to arbitrary token
    windows: a legal citation that cites half of Section 3(p) is worse than
    useless, and a missing section marker almost always means the source text
    needs manual cleanup, not a more forgiving chunker.
    """
    lines = full_text.split("\n")

    # Pass 1 -- find heading positions (line index, char offset, ref label).
    boundaries: list[tuple[int, str]] = []  # (char_offset, ref)
    char_offset = 0
    prev_blank = True  # treat document start as after a blank line

    for line in lines:
        ref, this_blank = _classify_line(line, prev_blank)
        if ref is not None:
            boundaries.append((char_offset, ref))
        prev_blank = this_blank
        char_offset += len(line) + 1  # +1 for the "\n" we split on

    if not boundaries:
        raise ValueError(
            f"No recognizable Section/Rule headings found in '{act_name}'. Check "
            "the source text formatting before ingesting -- either the cleanup "
            "did not preserve 'Section N.' / 'Rule N.' (inline or paragraph-start "
            "style) or this act uses a heading style the chunker does not yet know "
            "about. Do not silently fall back to token-window chunking for statutes."
        )

    # Pass 2 -- slice full_text between consecutive boundary offsets.
    chunks: list[dict] = []
    for i, (start, ref) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
        text = full_text[start:end].strip()
        if text:
            chunks.append({"section_ref": ref, "chunk_text": text})

    # Sanity check -- not a hard fail.  A suspiciously short chunk usually
    # means two headings landed with nothing real between them.
    for c in chunks:
        if len(c["chunk_text"]) < len(c["section_ref"]) + 15:
            print(
                f"WARNING [{act_name}] suspiciously short chunk for "
                f"{c['section_ref']!r} ({len(c['chunk_text'])} chars) -- "
                "check source formatting around this heading."
            )

    return chunks
