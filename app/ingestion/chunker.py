import re

# Stress-test findings fixed here (see chat writeup for full repros):
#
# 1. No line anchor at all -> matched cross-references anywhere in the text,
#    not just real headings (e.g. "as defined under Section 2(1)(l), which..."
#    embedded inside Section 3's own body). Fixed by requiring a genuine
#    heading position (see _INLINE_HEADING / _PARAGRAPH_HEADING below).
#
# 2. Hard-wrapped source text: a cross-reference number can coincidentally
#    land at the start of a *wrapped* line. A negative lookahead for a
#    trailing comma doesn't survive backtracking (the engine just shrinks
#    "2(1)(l)" to "2(1)(l" so the next char is ")" instead of ","). Fixed by
#    requiring punctuation immediately after the bare number/letter suffix
#    instead of trying to exclude commas.
#
# 3. Headings where the title is on the NEXT line ("Section 3\nWhat are not
#    inventions.-") were missed entirely by fix #2, because there's no
#    punctuation right after "Section 3" on that line. The reliable signal
#    here isn't punctuation - it's that a genuine heading starts a fresh
#    paragraph (blank line before it), while a wrapped cross-reference never
#    does (it's still mid-sentence, mid-paragraph). Added _PARAGRAPH_HEADING
#    to catch this: "Section N" / "Rule N" preceded by a blank line (or start
#    of document) and followed by nothing but whitespace before a newline.
#
# Both alternatives are ANDed with "must look like a real boundary, not just
# a number appearing somewhere" - that's the actual invariant being
# protected, not any one specific punctuation/line-break pattern. If your
# real corpus text uses a heading style neither alternative catches, this
# will raise ValueError below (fail loud, per this file's design) rather
# than silently mis-chunk - extend the pattern then, with a real example in
# hand, rather than loosening either alternative preemptively.
#
# ASSUMPTION TO VERIFY once real corpus text lands: this also assumes
# "Section N" / "Rule N" is the literal heading text (not bare "3."
# numbering, which is how some official gazette PDFs actually print it), and
# assumes Unix line endings (LF) - normalize \r\n to \n during corpus
# cleanup if the source files come from Windows-authored text.
_REF = r"(?:Section\s+\d+[A-Za-z]*|Rule\s+\d+[A-Za-z]*)"

# Alt A: heading text immediately followed by '.', em-dash, or '-' on the
# same line - the common case ("Section 3. What are not inventions.-").
_INLINE_HEADING = rf"^[ \t]*({_REF})(?=[ \t]*[.\u2014-])"

# Alt B: heading number stands alone on its own line, at the start of a
# paragraph (preceded by a blank line or start of document), with the title
# text following on the next line instead of immediately after.
_PARAGRAPH_HEADING = rf"(?:\A|(?<=\n\n))[ \t]*({_REF})(?=[ \t]*\n)"

SECTION_PATTERN = re.compile(
    f"{_INLINE_HEADING}|{_PARAGRAPH_HEADING}",
    re.IGNORECASE | re.MULTILINE,
)


def chunk_by_section(full_text: str, act_name: str) -> list[dict]:
    """Split statute text at Section/Rule boundaries.

    Deliberately raises instead of silently falling back to arbitrary token
    windows: a legal citation that cites half of Section 3(p) is worse than
    useless, and a missing section marker almost always means the source text
    needs manual cleanup, not a more forgiving chunker.
    """
    matches = list(SECTION_PATTERN.finditer(full_text))
    if not matches:
        raise ValueError(
            f"No recognizable Section/Rule headings found in '{act_name}'. Check "
            "the source text formatting before ingesting -- either the cleanup "
            "didn't preserve 'Section N.' / 'Rule N.' (inline or as its own "
            "paragraph-start line) or this act uses a different heading style "
            "that SECTION_PATTERN needs to be extended for. Do not silently "
            "fall back to token-window chunking for statutory text."
        )

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        # Alt A fills group 1, Alt B fills group 2 - exactly one will be set.
        section_ref = (m.group(1) or m.group(2)).strip()
        text = full_text[start:end].strip()
        if text:
            chunks.append({"section_ref": section_ref, "chunk_text": text})

    # Sanity check, not a hard fail: a suspiciously short chunk usually means
    # two headings landed with nothing real between them - worth a human
    # glance, not worth blocking ingestion over.
    for c in chunks:
        if len(c["chunk_text"]) < len(c["section_ref"]) + 15:
            print(
                f"WARNING [{act_name}] suspiciously short chunk for "
                f"{c['section_ref']!r} ({len(c['chunk_text'])} chars) -- "
                "check source formatting around this heading."
            )

    return chunks
