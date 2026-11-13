import logging

from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)


def parse_office(file_path: str):
    """
    Parses Office documents (.docx, .pptx) using the Unstructured library.
    Unlike PDFs, these formats are structured and lightweight, so they are processed locally.
    """
    try:
        elements = partition(filename=file_path)
        full_text = "\n".join([str(el) for el in elements])

        if not full_text.strip():
            logger.warning("Unstructured returned empty text for %s", file_path)
        else:
            logger.info("Parsed %d characters from %s", len(full_text), file_path)

        return full_text
    except Exception:
        logger.exception("Office document parsing failed for %s", file_path)
        raise