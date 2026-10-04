from pypdf import PdfReader


# =========================================================
# PDF PAGE EXTRACTION
# =========================================================

def extract_pages_from_pdf(uploaded_file):
    """
    Extract PDF text page-by-page.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    if uploaded_file is None:
        raise ValueError(
            "No PDF file was provided."
        )

    try:
        reader = PdfReader(uploaded_file)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text()

            if text and text.strip():

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text.strip(),
                    }
                )

        if not pages:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        return pages

    except Exception as error:
        raise RuntimeError(
            f"Could not read the PDF: {error}"
        )


# =========================================================
# BACKWARD-COMPATIBLE FULL TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract the complete PDF text.

    This function is kept for compatibility with
    existing application code.

    Page boundaries are preserved with blank lines.
    """

    pages = extract_pages_from_pdf(
        uploaded_file
    )

    page_texts = []

    for page in pages:

        page_texts.append(
            page["text"]
        )

    full_text = "\n\n".join(
        page_texts
    ).strip()

    if not full_text:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    return full_text


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(
    pages_or_text,
    chunk_size=800,
    chunk_overlap=100,
):
    """
    Split PDF content into searchable chunks while
    preserving page information.

    Input can be:

    1. Page records returned by extract_pages_from_pdf()

    OR

    2. A normal string.

    Returns:

        [
            {
                "text": "...",
                "page_number": 1,
                "chunk_number": 1
            }
        ]
    """

    if not pages_or_text:
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )


    # =====================================================
    # NORMALIZE INPUT
    # =====================================================

    if isinstance(
        pages_or_text,
        str,
    ):

        pages = [
            {
                "page_number": None,
                "text": pages_or_text,
            }
        ]

    elif isinstance(
        pages_or_text,
        list,
    ):

        pages = pages_or_text

    else:

        raise ValueError(
            "Unsupported PDF text format."
        )


    # =====================================================
    # CREATE CHUNKS
    # =====================================================

    chunks = []

    global_chunk_number = 1

    for page in pages:

        page_number = page.get(
            "page_number"
        )

        text = page.get(
            "text",
            "",
        )

        if not text or not text.strip():
            continue

        text = text.strip()

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length,
            )

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page_number": page_number,
                        "chunk_number": global_chunk_number,
                    }
                )

                global_chunk_number += 1

            if end >= text_length:
                break

            start = (
                end - chunk_overlap
            )

    return chunks


# =========================================================
# GET CHUNK TEXT
# =========================================================

def get_chunk_text(chunk):
    """
    Safely get text from either:

    - New chunk dictionary
    - Old string chunk
    """

    if isinstance(
        chunk,
        dict,
    ):
        return chunk.get(
            "text",
            "",
        )

    if isinstance(
        chunk,
        str,
    ):
        return chunk

    return ""


# =========================================================
# GET PAGE NUMBER
# =========================================================

def get_chunk_page(chunk):
    """
    Return the page number of a chunk.
    """

    if isinstance(
        chunk,
        dict,
    ):
        return chunk.get(
            "page_number"
        )

    return None
