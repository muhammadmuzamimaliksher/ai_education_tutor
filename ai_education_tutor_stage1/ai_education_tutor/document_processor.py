from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF file.

    Parameters:
        uploaded_file: Streamlit uploaded PDF file

    Returns:
        Extracted text as a string
    """

    if uploaded_file is None:
        raise ValueError("No PDF file was provided.")

    try:
        reader = PdfReader(uploaded_file)

        pages_text = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                pages_text.append(text)

        full_text = "\n".join(pages_text).strip()

        if not full_text:
            raise ValueError(
                "No readable text was found in this PDF."
            )

        return full_text

    except Exception as error:
        raise RuntimeError(
            f"Could not read the PDF: {error}"
        )


def split_text(
    text,
    chunk_size=800,
    chunk_overlap=100,
):
    """
    Split document text into overlapping chunks.

    Parameters:
        text: Full document text
        chunk_size: Approximate characters per chunk
        chunk_overlap: Characters shared between chunks

    Returns:
        List of text chunks
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = text.strip()

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks
