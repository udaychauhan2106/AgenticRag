from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.config import settings
from app.pages.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractionMetadata,
    ExtractionMethod,
    PageContent,
    PageIngestRequest,
    PageSource,
)
from app.pages.validators import (
    validate_content_blocks,
    validate_content_size,
    validate_extraction_metadata,
    validate_page_content,
    validate_page_url,
)


def _content(text: str = "A page.", block_count: int = 1) -> PageContent:
    return PageContent(
        text=text,
        blocks=[
            ContentBlock(type=ContentBlockType.paragraph, text=f"block {index}")
            for index in range(block_count)
        ],
    )


def _extraction(extracted_at: datetime, word_count: int = 2) -> ExtractionMetadata:
    return ExtractionMetadata(
        method=ExtractionMethod.fallback,
        extracted_at=extracted_at,
        word_count=word_count,
        was_truncated=False,
    )


def _request(content: PageContent, extracted_at: datetime) -> PageIngestRequest:
    return PageIngestRequest(
        schema_version="1.0",
        source=PageSource(url="https://example.com", title="Example"),
        content=content,
        extraction=_extraction(extracted_at),
    )


@pytest.mark.parametrize("url", ["http://example.com/path", "https://example.com/path"])
def test_validate_page_url_accepts_http_and_https(url: str) -> None:
    validated = validate_page_url(url)

    assert str(validated).startswith(url.split(":")[0] + "://")


def test_validate_page_url_rejects_unsupported_scheme() -> None:
    with pytest.raises(ValueError, match="HTTP or HTTPS"):
        validate_page_url("ftp://example.com/file")


def test_validate_page_url_rejects_embedded_credentials() -> None:
    with pytest.raises(ValueError, match="credentials"):
        validate_page_url("https://user:secret@example.com/")


@pytest.mark.parametrize("url", ["http:///missing-host", "http://exa mple.com"])
def test_validate_page_url_rejects_malformed_or_missing_hostname(url: str) -> None:
    with pytest.raises(ValueError, match="hostname|malformed"):
        validate_page_url(url)


def test_page_schema_rejects_empty_webpage_content() -> None:
    with pytest.raises(ValidationError, match="page content must not be empty"):
        PageContent(text=" \t\n ", blocks=[ContentBlock(type="paragraph", text="block")])


def test_validate_content_size_measures_utf8_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LIVERAG_MAX_EXTRACTED_TEXT_BYTES", 4)
    content = _content(text="éé")

    with pytest.raises(ValueError, match="UTF-8 bytes"):
        validate_content_size(content)


def test_validate_content_size_checks_structured_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LIVERAG_MAX_EXTRACTED_TEXT_BYTES", 8)
    content = PageContent(
        text="small",
        blocks=[
            ContentBlock(type="paragraph", text="four"),
            ContentBlock(type="paragraph", text="five"),
        ],
    )

    with pytest.raises(ValueError, match="structured block text"):
        validate_content_size(content)


def test_validate_content_blocks_enforces_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LIVERAG_MAX_CONTENT_BLOCKS", 2)

    with pytest.raises(ValueError, match="maximum is 2"):
        validate_content_blocks(_content(block_count=3))


def test_validate_content_blocks_rejects_unmeaningful_constructed_block() -> None:
    content = PageContent.model_construct(
        text="page",
        blocks=[ContentBlock.model_construct(type=ContentBlockType.paragraph, text=" \t")],
    )

    with pytest.raises(ValueError, match="meaningful text"):
        validate_content_blocks(content)


def test_validate_extraction_metadata_accepts_recent_aware_timestamp() -> None:
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)

    validate_extraction_metadata(_extraction(now - timedelta(days=1)), now=now)


def test_validate_extraction_metadata_rejects_timestamp_beyond_clock_skew() -> None:
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="clock-skew"):
        validate_extraction_metadata(
            _extraction(now + timedelta(minutes=6)),
            now=now,
        )


def test_validate_extraction_metadata_rejects_unreasonably_old_timestamp() -> None:
    now = datetime(2026, 1, 31, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="maximum accepted age"):
        validate_extraction_metadata(
            _extraction(now - timedelta(days=31)),
            now=now,
        )


def test_client_word_count_is_not_used_as_authoritative_count() -> None:
    from app.pages.normalizer import prepare_page

    now = datetime.now(timezone.utc)
    request = _request(
        PageContent(
            text="client says many words",
            blocks=[ContentBlock(type="paragraph", text="backend count")],
        ),
        now,
    )
    request.extraction.word_count = 999

    validate_page_content(request)
    prepared = prepare_page(request)

    assert request.extraction.word_count == 999
    assert prepared.word_count == 2
