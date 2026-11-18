
import re
from datetime import datetime
from enum import Enum
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)



class StrictBaseModel(BaseModel):
    """
    Base model.

    Rejects unexpected fields instead of silently ignoring them.
    """

    model_config = ConfigDict(extra="forbid")



class ContentBlockType(str, Enum):
    """Kinds of structured text blocks extracted from a webpage."""

    heading = "heading"
    paragraph = "paragraph"
    list_item = "list_item"
    code = "code"
    quote = "quote"
    table = "table"


class ExtractionMethod(str, Enum):
    """Method used by the client to extract webpage content."""

    readability = "readability"
    main_element = "main_element"
    article_element = "article_element"
    fallback = "fallback"


class PageStatus(str, Enum):
    """Lifecycle status of a temporarily ingested webpage."""

    received = "received"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


def normalize_optional_string(value: str | None) -> str | None:
    """
    Strip optional metadata strings.

    Converts empty or whitespace-only strings to None.
    """

    if value is None:
        return None

    normalized = value.strip()

    return normalized if normalized else None


def validate_aware_datetime(value: datetime) -> datetime:
    """
    Ensure a datetime contains timezone information.

    Naive datetimes are rejected to avoid ambiguous timestamps.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")

    return value


def validate_expiry(
    created_at: datetime,
    expires_at: datetime,
) -> None:
    """
    Ensure expiry occurs after creation.

    Both timestamps must be timezone-aware.
    """

    if expires_at <= created_at:
        raise ValueError("expires_at must be later than created_at")



class PageSource(StrictBaseModel):
    """URL and metadata reported by the extracting client."""

    url: HttpUrl
    title: str

    description: str | None = None
    author: str | None = None
    site_name: str | None = None
    published_at: datetime | None = None
    language: str | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        """Reject empty titles and remove surrounding whitespace."""

        value = value.strip()

        if not value:
            raise ValueError("title must not be empty or whitespace-only")

        return value

    @field_validator(
        "description",
        "author",
        "site_name",
        "language",
    )
    @classmethod
    def normalize_optional_fields(
        cls,
        value: str | None,
    ) -> str | None:
        return normalize_optional_string(value)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is not None:
            return validate_aware_datetime(value)

        return None



class ContentBlock(StrictBaseModel):
    """A structured text block extracted from a webpage."""

    type: ContentBlockType
    text: str

    level: int | None = None
    language: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """
        Reject empty blocks.

        Preserve the original whitespace because indentation
        may be important in code blocks.
        """

        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")

        return value

    @field_validator("language")
    @classmethod
    def normalize_language(
        cls,
        value: str | None,
    ) -> str | None:
        return normalize_optional_string(value)

    @model_validator(mode="after")
    def validate_heading_level(self) -> Self:
        """Only heading blocks may have a heading level."""

        if self.type == ContentBlockType.heading:
            if self.level is None or not 1 <= self.level <= 6:
                raise ValueError(
                    "heading level must be between 1 and 6"
                )

        elif self.level is not None:
            raise ValueError(
                "level must be absent for non-heading blocks"
            )

        return self



class PageContent(StrictBaseModel):
    """Full extracted page text and structured content blocks."""

    text: str

    blocks: list[ContentBlock] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Reject empty webpage content."""

        if not value.strip():
            raise ValueError(
                "page content must not be empty or whitespace-only"
            )

        return value



class ExtractionMetadata(StrictBaseModel):
    """Client-reported information about webpage extraction."""

    method: ExtractionMethod
    extracted_at: datetime

    word_count: int = Field(ge=0)

    was_truncated: bool

    @field_validator("extracted_at")
    @classmethod
    def validate_extracted_at(cls, value: datetime) -> datetime:
        return validate_aware_datetime(value)



class PageIngestRequest(StrictBaseModel):
    """
    Request sent by the Chrome extension to ingest a webpage.

    Backend-generated fields such as page_id and expires_at
    are deliberately excluded.
    """

    schema_version: Literal["1.0"]

    source: PageSource
    content: PageContent
    extraction: ExtractionMetadata



class PageSourceSummary(StrictBaseModel):
    """Compact page identity included in API responses."""

    url: HttpUrl
    title: str

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("title must not be empty or whitespace-only")

        return value



class PageIngestResponse(StrictBaseModel):
    """
    Backend-generated result of a page ingestion operation.
    """

    page_id: UUID

    status: PageStatus

    source: PageSourceSummary

    content_hash: str

    chunk_count: int = Field(ge=0)

    created_at: datetime
    expires_at: datetime

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        """Validate a 64-character SHA-256 hexadecimal digest."""

        if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError(
                "content_hash must be a 64-character "
                "SHA-256 hexadecimal digest"
            )

        return value.lower()

    @field_validator("created_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        return validate_aware_datetime(value)

    @model_validator(mode="after")
    def validate_expiry(self) -> Self:
        validate_expiry(self.created_at, self.expires_at)

        return self




class PageStatusResponse(StrictBaseModel):
    """Backend-reported status and expiry of an ingested page."""

    page_id: UUID

    status: PageStatus

    chunk_count: int = Field(ge=0)

    created_at: datetime
    expires_at: datetime

    error: str | None = None

    @field_validator("created_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        return validate_aware_datetime(value)

    @field_validator("error")
    @classmethod
    def normalize_error(
        cls,
        value: str | None,
    ) -> str | None:
        return normalize_optional_string(value)

    @model_validator(mode="after")
    def validate_expiry(self) -> Self:
        validate_expiry(self.created_at, self.expires_at)

        return self
