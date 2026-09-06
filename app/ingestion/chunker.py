import re

SECTION_PATTERN = re.compile(r"(Section\s+\d+[A-Za-z()\-]*|Rule\s+\d+[A-Za-z]*)", re.IGNORECASE)


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
            f"No Section/Rule boundaries found in '{act_name}'. Check the source "
            "text formatting before ingesting - do not silently fall back to "
            "token-window chunking for statutory text."
        )

    chunks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_ref = m.group(1).strip()
        text = full_text[start:end].strip()
        if text:
            chunks.append({"section_ref": section_ref, "chunk_text": text})
    return chunks
