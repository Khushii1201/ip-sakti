import re

# Stress-test finding: the original pattern had no line anchor, so it matched
# "Section 3(d)" or "Rule 158B" ANYWHERE in the text -- including inline
# cross-references inside a totally different section's body (e.g. "as
# defined under Section 2(1)(l)" appearing mid-sentence in Section 3's own
# text). That fragments the real section and mis-attributes its tail to a
# garbled section_ref. Confirmed by reproducing it against a real
# cross-reference pattern lifted from the Patents Act (Section 3(d)).
#
# A second, narrower case the line-anchor alone doesn't catch: hard-wrapped
# source text where a cross-reference number happens to land at the start of
# a *wrapped* line by coincidence (e.g. "...as defined under\nSection 2(1)(l),
# which does not..."). Tried a negative lookahead for a trailing comma first
# (rejecting matches immediately followed by ","), but the greedy character
# class just backtracks around it -- shrinking "2(1)(l)" down to "2(1)(l"
# so the next character is ")" instead of ",", which slips past the
# lookahead. That's a real regex footgun, not a hypothetical one; it showed
# up in testing.
#
# The fix that actually holds up: stop trying to capture nested parenthetical
# sub-references in the heading itself, and instead require punctuation
# ('.', em-dash, or '-') immediately (allowing only whitespace) after the
# bare number/letter suffix -- which is how genuine headings actually read
# ("Section 3. What are not inventions.-", "Rule 158B."). A cross-reference
# is never followed directly by that punctuation ("Section 2(1)(l), which..."
# hits "(" first; "Section 3 above the applicant" hits a bare word first),
# so it can no longer be shrunk into a false match. The trade-off: a heading
# that puts its title on the next line instead of right after the number
# won't be caught -- verify against your actual cleaned corpus text and
# extend this pattern if that turns out to be how it's formatted, rather
# than loosening the punctuation requirement back to bare whitespace.
#
# ASSUMPTION TO VERIFY once real corpus text lands: this also assumes
# "Section N" / "Rule N" is the literal heading text (not bare "3."
# numbering, which is how some official gazette PDFs actually print it). If
# the corpus person's cleanup produces bare numeric headings instead, this
# pattern will raise ValueError below (fail loud, per this file's own
# design) rather than silently mis-chunk.
SECTION_PATTERN = re.compile(
    r"^\s*(Section\s+\d+[A-Za-z]*|Rule\s+\d+[A-Za-z]*)(?=\s*[.\u2014-])",
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
            f"No line-anchored Section/Rule headings found in '{act_name}'. Check "
            "the source text formatting before ingesting -- either the cleanup "
            "didn't preserve 'Section N.' / 'Rule N.' at the start of each "
            "section's line, or this act uses a different heading style that "
            "SECTION_PATTERN needs to be extended for. Do not silently fall "
            "back to token-window chunking for statutory text."
        )

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_ref = m.group(1).strip()
        text = full_text[start:end].strip()
        if text:
            chunks.append({"section_ref": section_ref, "chunk_text": text})

    # Sanity check, not a hard fail: a suspiciously short chunk usually means
    # two headings landed with nothing real between them -- worth a human
    # glance, not worth blocking ingestion over.
    for c in chunks:
        if len(c["chunk_text"]) < len(c["section_ref"]) + 15:
            print(
                f"WARNING [{act_name}] suspiciously short chunk for "
                f"{c['section_ref']!r} ({len(c['chunk_text'])} chars) -- "
                "check source formatting around this heading."
            )

    return chunks
