"""Application-level validation for LiveRAG page ingestion requests."""

from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from pydantic import HttpUrl, TypeAdapter, ValidationError

from app.config import settings
from app.pages.schemas import (
    ExtractionMetadata,
    PageContent,
    PageIngestRequest,
)

_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


def validate_page_url(url: HttpUrl | str) -> HttpUrl:
    """Validate an HTTP(S) page URL without fetching it."""
    raw_url = str(url)
    try:
        parsed = urlsplit(raw_url)
    except ValueError as exc:
        raise ValueError("page URL is malformed") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("page URL must use HTTP or HTTPS")
    if not parsed.hostname:
        raise ValueError("page URL must include a valid hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("page URL must not include embedded credentials")

    try:
        parsed.port
        return _HTTP_URL_ADAPTER.validate_python(raw_url)
    except (ValueError, ValidationError) as exc:
        raise ValueError("page URL is malformed or has an invalid hostname") from exc


def validate_content_size(content: PageContent) -> None:
    """Enforce the configured UTF-8 size limit on text and structured blocks."""
    if not isinstance(content, PageContent):
        raise TypeError("content must be a PageContent model")

    maximum = settings.LIVERAG_MAX_EXTRACTED_TEXT_BYTES
    text_size = len(content.text.encode("utf-8"))
    block_text_size = sum(len(block.text.encode("utf-8")) for block in content.blocks)
    if len(content.blocks) > 1:
        block_text_size += len(content.blocks) - 1

    if text_size > maximum:
        raise ValueError(
            f"extracted text is {text_size} UTF-8 bytes; maximum is {maximum} bytes"
        )
    if block_text_size > maximum:
        raise ValueError(
            f"structured block text is {block_text_size} UTF-8 bytes; "
            f"maximum is {maximum} bytes"
        )


def validate_content_blocks(content: PageContent) -> None:
    """Enforce the configured block count and meaningful-text requirements."""
    if not isinstance(content, PageContent):
        raise TypeError("content must be a PageContent model")

    maximum = settings.LIVERAG_MAX_CONTENT_BLOCKS
    if len(content.blocks) > maximum:
        raise ValueError(
            f"content has {len(content.blocks)} blocks; maximum is {maximum}"
        )

    for index, block in enumerate(content.blocks):
        if not block.text.strip():
            raise ValueError(f"content block {index} must contain meaningful text")


def validate_extraction_metadata(
    extraction: ExtractionMetadata,
    *,
    now: datetime | None = None,
) -> None:
    """Validate timestamp policy; client-reported word count remains untrusted."""
    if not isinstance(extraction, ExtractionMetadata):
        raise TypeError("extraction must be an ExtractionMetadata model")

    extracted_at = extraction.extracted_at
    if extracted_at.tzinfo is None or extracted_at.utcoffset() is None:
        raise ValueError("extracted_at must be timezone-aware")

    reference_time = now or datetime.now(timezone.utc)
    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        raise ValueError("validation time must be timezone-aware")

    latest_allowed = reference_time + timedelta(
        seconds=settings.LIVERAG_EXTRACTION_MAX_FUTURE_SKEW_SECONDS
    )
    if extracted_at > latest_allowed:
        raise ValueError(
            "extracted_at is in the future beyond the allowed clock-skew tolerance"
        )

    oldest_allowed = reference_time - timedelta(
        days=settings.LIVERAG_EXTRACTION_MAX_AGE_DAYS
    )
    if extracted_at < oldest_allowed:
        raise ValueError("extracted_at is older than the maximum accepted age")


def validate_page_content(request: PageIngestRequest) -> None:
    """Run URL, content-size, block-structure, and metadata policy checks."""
    if not isinstance(request, PageIngestRequest):
        raise TypeError("request must be a PageIngestRequest model")

    validate_page_url(request.source.url)
    validate_content_size(request.content)
    validate_content_blocks(request.content)
    validate_extraction_metadata(request.extraction)
