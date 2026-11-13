from typing import List


def chunk_text(text: str, chunk_size:int=1500)->List[str]:
    """
    Simple chunker that splits by paragraphs 
    Ensure chunks do not exceed the specified size.
    """
    if not text.strip():
        return []
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if chunk.strip()]
    
        