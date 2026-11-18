"""Deterministic normalization and preparation of LiveRAG page content."""

import hashlib
import re
import unicodedata

from pydantic import BaseModel, ConfigDict

from app.pages.schemas import (
    ContentBlock,
    ContentBlockType,
    PageIngestRequest,
    PageSource,
)
from app.pages.validators import validate_page_content

_PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n+")
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_WORD = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", re.UNICODE)
_BACKTICK_RUN = re.compile(r"`+")


class PreparedPage(BaseModel):
    """Backend-prepared page content, separate from public API schemas."""

    model_config = ConfigDict(extra="forbid")

    source: PageSource
    normalized_text: str
    normalized_blocks: list[ContentBlock]
    content_hash: str
    word_count: int


def _normalize_line_endings(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def _normalize_inline_whitespace(text: str) -> str:
    return _HORIZONTAL_WHITESPACE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Normalize prose to NFC, single spaces, and explicit paragraph breaks."""
    normalized = _normalize_line_endings(text).strip()
    if not normalized:
        return ""

    paragraphs = _PARAGRAPH_BREAK.split(normalized)
    normalized_paragraphs = []
    for paragraph in paragraphs:
        flattened = paragraph.replace("\n", " ")
        normalized_paragraph = _normalize_inline_whitespace(flattened)
        if normalized_paragraph:
            normalized_paragraphs.append(normalized_paragraph)

    return "\n\n".join(normalized_paragraphs)


def _normalize_table_text(text: str) -> str:
    """Normalize table cells while preserving tab-delimited columns and rows."""
    normalized = _normalize_line_endings(text)
    rows = []
    for row in normalized.split("\n"):
        cells = row.split("\t")
        rows.append(
            "\t".join(_normalize_inline_whitespace(cell) for cell in cells)
        )
    return "\n".join(rows)


def normalize_blocks(blocks: list[ContentBlock]) -> list[ContentBlock]:
    """Return normalized block copies while retaining type and metadata."""
    normalized_blocks: list[ContentBlock] = []
    for block in blocks:
        if not isinstance(block, ContentBlock):
            raise TypeError("blocks must contain ContentBlock models")

        if block.type == ContentBlockType.code:
            normalized_text = _normalize_line_endings(block.text)
        elif block.type == ContentBlockType.table:
            normalized_text = _normalize_table_text(block.text)
        else:
            normalized_text = normalize_text(block.text)

        normalized_blocks.append(block.model_copy(update={"text": normalized_text}))

    return normalized_blocks


def _render_code_block(block: ContentBlock) -> str:
    longest_backticks = max(
        (len(match.group()) for match in _BACKTICK_RUN.finditer(block.text)),
        default=0,
    )
    fence = "`" * max(3, longest_backticks + 1)
    language = block.language or ""
    trailing_newline = "" if block.text.endswith("\n") else "\n"
    return f"{fence}{language}\n{block.text}{trailing_newline}{fence}"


def build_normalized_text(blocks: list[ContentBlock]) -> str:
    """Render normalized blocks with explicit boundaries and structure."""
    rendered_blocks = []
    for block in blocks:
        if not isinstance(block, ContentBlock):
            raise TypeError("blocks must contain ContentBlock models")

        if block.type == ContentBlockType.heading:
            rendered = f"{'#' * block.level} {block.text}"
        elif block.type == ContentBlockType.list_item:
            rendered = "- " + block.text.replace("\n", "\n  ")
        elif block.type == ContentBlockType.quote:
            rendered = "\n".join(f"> {line}" for line in block.text.split("\n"))
        elif block.type == ContentBlockType.code:
            rendered = _render_code_block(block)
        elif block.type == ContentBlockType.table:
            rendered = f"Table:\n{block.text}"
        else:
            rendered = block.text
        rendered_blocks.append(rendered)

    return "\n\n".join(rendered_blocks)


def calculate_content_hash(normalized_text: str) -> str:
    """Return the SHA-256 hexadecimal digest of normalized UTF-8 content."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def count_words(normalized_text: str) -> int:
    """Count Unicode word-like tokens; CJK segmentation and code token counts vary."""
    return len(_WORD.findall(normalized_text))


def prepare_page(request: PageIngestRequest) -> PreparedPage:
    """Validate and deterministically prepare a request for downstream processing."""
    if not isinstance(request, PageIngestRequest):
        raise TypeError("request must be a PageIngestRequest model")

    validate_page_content(request)
    normalized_blocks = normalize_blocks(request.content.blocks)
    normalized_text = build_normalized_text(normalized_blocks)

    return PreparedPage(
        source=request.source,
        normalized_text=normalized_text,
        normalized_blocks=normalized_blocks,
        content_hash=calculate_content_hash(normalized_text),
        word_count=count_words(normalized_text),
    )
