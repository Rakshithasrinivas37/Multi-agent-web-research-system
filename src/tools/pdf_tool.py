"""PDF download and text extraction tool."""

import io

import httpx

from src.tools.text_utils import clean_text


def extract_pdf_text(url: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("pypdf package is not installed") from error

    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    reader = PdfReader(io.BytesIO(response.content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def is_pdf_url(url: str) -> bool:
    return url.lower().split("?", 1)[0].endswith(".pdf")
