import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """
    Extract text from a PDF locally using pypdf.
    Falls back to pdfplumber for pages that yield no text (e.g. image-heavy pages).
    """
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info("PDF has %d pages: %s", total_pages, file_path)

        text_parts: list[str] = []
        blank_pages: list[int] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                text_parts.append(text)
            else:
                blank_pages.append(i + 1)

        if blank_pages:
            logger.info("pypdf returned blank pages %s; retrying with pdfplumber", blank_pages)
            try:
                import pdfplumber

                with pdfplumber.open(file_path) as pdf:
                    for page_num in blank_pages:
                        page = pdf.pages[page_num - 1]
                        fallback_text = page.extract_text() or ""
                        if fallback_text.strip():
                            text_parts.append(fallback_text)
            except Exception:
                logger.exception("pdfplumber fallback failed for %s", file_path)

        full_text = "\n".join(text_parts)

        if not full_text.strip():
            logger.warning("No text extracted from %s; file may be image-based", file_path)
        else:
            logger.info("Extracted %d characters from %s", len(full_text), file_path)

        return full_text
    except Exception:
        logger.exception("PDF parsing failed for %s", file_path)
        raise