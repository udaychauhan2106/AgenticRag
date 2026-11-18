from datetime import datetime, timezone

from app.pages.normalizer import (
    build_normalized_text,
    calculate_content_hash,
    count_words,
    normalize_blocks,
    normalize_text,
    prepare_page,
)
from app.pages.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractionMetadata,
    ExtractionMethod,
    PageContent,
    PageIngestRequest,
    PageSource,
)


def _block(
    block_type: ContentBlockType,
    text: str,
    *,
    level: int | None = None,
    language: str | None = None,
) -> ContentBlock:
    return ContentBlock(type=block_type, text=text, level=level, language=language)


def _request(
    blocks: list[ContentBlock],
    *,
    text: str = "This full text is deliberately not authoritative.",
) -> PageIngestRequest:
    return PageIngestRequest(
        schema_version="1.0",
        source=PageSource(url="https://example.com", title="Example"),
        content=PageContent(text=text, blocks=blocks),
        extraction=ExtractionMetadata(
            method=ExtractionMethod.fallback,
            extracted_at=datetime.now(timezone.utc),
            word_count=1000,
            was_truncated=False,
        ),
    )


def test_normalize_text_preserves_multiple_paragraphs_and_collapses_spacing() -> None:
    result = normalize_text("  First   paragraph.\r\nwrapped\r\n\r\n  Second\tparagraph.  ")

    assert result == "First paragraph. wrapped\n\nSecond paragraph."


def test_normalize_text_handles_windows_and_unix_line_endings_identically() -> None:
    assert normalize_text("One\r\nTwo\r\n\r\nThree") == normalize_text(
        "One\nTwo\n\nThree"
    )
    assert normalize_text("One\rTwo\r\rThree") == "One Two\n\nThree"


def test_normalize_text_uses_nfc_and_preserves_non_english_characters() -> None:
    result = normalize_text("Cafe\u0301 — 東京 — مرحبا — Привет")

    assert result == "Café — 東京 — مرحبا — Привет"


def test_build_normalized_text_preserves_heading_hierarchy() -> None:
    blocks = normalize_blocks(
        [
            _block(ContentBlockType.heading, " Main title ", level=1),
            _block(ContentBlockType.heading, " Subheading ", level=3),
            _block(ContentBlockType.paragraph, "Body text."),
        ]
    )

    assert build_normalized_text(blocks) == (
        "# Main title\n\n### Subheading\n\nBody text."
    )


def test_code_normalization_preserves_indentation_spaces_and_language() -> None:
    block = _block(
        ContentBlockType.code,
        "    def f():\r\n        return  x  \r\n",
        language=" python ",
    )
    normalized = normalize_blocks([block])[0]

    assert normalized.text == "    def f():\n        return  x  \n"
    assert normalized.language == "python"
    assert build_normalized_text([normalized]) == (
        "```python\n    def f():\n        return  x  \n```"
    )


def test_list_items_remain_separate_and_ordered() -> None:
    blocks = normalize_blocks(
        [
            _block(ContentBlockType.list_item, " First item "),
            _block(ContentBlockType.list_item, " Second item "),
        ]
    )

    assert build_normalized_text(blocks) == "- First item\n\n- Second item"


def test_table_normalization_preserves_rows_and_tab_columns() -> None:
    block = _block(
        ContentBlockType.table,
        " Name \t Value \r\nalpha\t 1\r\nbeta\t 2",
    )
    normalized = normalize_blocks([block])[0]

    assert normalized.text == "Name\tValue\nalpha\t1\nbeta\t2"
    assert build_normalized_text([normalized]) == (
        "Table:\nName\tValue\nalpha\t1\nbeta\t2"
    )


def test_normalized_hashes_are_stable_and_sensitive_to_meaningful_changes() -> None:
    first = normalize_text("A   useful paragraph.")
    equivalent = normalize_text("A useful paragraph.")
    changed = normalize_text("A different paragraph.")

    assert calculate_content_hash(first) == calculate_content_hash(equivalent)
    assert calculate_content_hash(first) != calculate_content_hash(changed)
    assert len(calculate_content_hash(first)) == 64


def test_prepare_page_builds_text_from_normalized_blocks_only() -> None:
    blocks = [
        _block(ContentBlockType.heading, "  Title  ", level=2),
        _block(ContentBlockType.paragraph, "  Actual   body.  "),
    ]
    request = _request(blocks, text="Unrelated duplicate source text.")

    prepared = prepare_page(request)

    assert prepared.normalized_text == "## Title\n\nActual body."
    assert prepared.normalized_text == build_normalized_text(prepared.normalized_blocks)
    assert prepared.content_hash == calculate_content_hash(prepared.normalized_text)
    assert prepared.word_count == count_words(prepared.normalized_text)
    assert request.content.blocks[0].text == "  Title  "


def test_prepare_page_is_deterministic_for_same_request_content() -> None:
    request = _request(
        [
            _block(ContentBlockType.heading, "Page title", level=1),
            _block(ContentBlockType.paragraph, "Some content."),
        ]
    )

    first = prepare_page(request)
    second = prepare_page(request)

    assert first.normalized_text == second.normalized_text
    assert first.content_hash == second.content_hash
    assert first.word_count == second.word_count
