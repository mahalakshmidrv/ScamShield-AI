"""File-upload validation. Screenshot OCR is NOT implemented in this MVP (clearly labelled extension)."""
import os

ALLOWED_EXT = {".png", ".jpg", ".jpeg"}
MAX_BYTES = 2 * 1024 * 1024
OCR_AVAILABLE = False  # flip only after a reliable Tesseract/EasyOCR integration is added and tested


class UploadError(ValueError):
    pass


def validate_upload(filename: str, size: int) -> None:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}.")
    if size > MAX_BYTES:
        raise UploadError(f"File too large ({size // 1024} KB). Limit is {MAX_BYTES // 1024} KB.")
    if not OCR_AVAILABLE:
        raise UploadError("Screenshot OCR is an extension that is not enabled in this build. Please paste the message text instead.")
