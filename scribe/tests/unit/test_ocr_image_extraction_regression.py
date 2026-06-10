# scribe/tests/unit/test_ocr_image_extraction_regression.py
"""
Regression tests for OCR processor image handling.

Fix 2: JPG/PNG files were sent as PDF document blocks to Claude Bedrock,
causing 400 Bad Request: 'The PDF specified was not valid.'

Fix: Detect image files by extension and send as image blocks instead.
"""
from pathlib import Path
from unittest.mock import Mock, patch

from django.test import TestCase, override_settings


@override_settings(
    PDF_EXTRACTION_STRATEGY="claude",
    PDF_EXTRACTION_MODEL="eu.anthropic.claude-sonnet-4-6",
    PDF_EXTRACTION_MAX_TOKENS=32000,
    TESSDATA_DIR="/tmp/tessdata",
    DEFAULT_MODEL_SCRIBE_OCR_PROCESSOR="claude-sonnet-4-6",
)
class TestImageFileDetection(TestCase):
    """OCRProcessor must detect image files and route them to _claude_extract_image."""

    def _get_processor(self):
        with patch("scribe.tools.ocr_processor.get_llm_client", side_effect=Exception("no LLM")):
            from scribe.tools.ocr_processor import OCRProcessor, OCRSettings

            return OCRProcessor(OCRSettings(tessdata_dir="/tmp/tessdata"))

    def test_image_media_types_mapping(self):
        """IMAGE_MEDIA_TYPES must include common image formats."""
        from scribe.tools.ocr_processor import OCRProcessor

        assert ".jpg" in OCRProcessor.IMAGE_MEDIA_TYPES
        assert ".jpeg" in OCRProcessor.IMAGE_MEDIA_TYPES
        assert ".png" in OCRProcessor.IMAGE_MEDIA_TYPES
        assert ".gif" in OCRProcessor.IMAGE_MEDIA_TYPES
        assert ".webp" in OCRProcessor.IMAGE_MEDIA_TYPES

        assert OCRProcessor.IMAGE_MEDIA_TYPES[".jpg"] == "image/jpeg"
        assert OCRProcessor.IMAGE_MEDIA_TYPES[".png"] == "image/png"

    def test_process_pdf_to_markdown_routes_jpg_to_image_extraction(self):
        """JPG files must be routed to _claude_extract_image, not _claude_extract_pdf."""
        processor = self._get_processor()

        with (
            patch.object(processor, "_claude_extract_image", return_value="<!-- PAGE 1 -->\nTest") as mock_image,
            patch.object(processor, "_claude_extract_pdf") as mock_pdf,
            patch.object(processor, "_pymupdf4llm_extract") as mock_fallback,
        ):
            result = processor._process_pdf_to_markdown(Path("/tmp/test.jpg"), "/tmp/images")

        mock_image.assert_called_once_with(Path("/tmp/test.jpg"))
        mock_pdf.assert_not_called()
        mock_fallback.assert_not_called()
        assert "PAGE 1" in result

    def test_process_pdf_to_markdown_routes_png_to_image_extraction(self):
        """PNG files must also be routed to _claude_extract_image."""
        processor = self._get_processor()

        with (
            patch.object(processor, "_claude_extract_image", return_value="<!-- PAGE 1 -->\nChart") as mock_image,
            patch.object(processor, "_claude_extract_pdf") as mock_pdf,
        ):
            processor._process_pdf_to_markdown(Path("/tmp/chart.png"), "/tmp/images")

        mock_image.assert_called_once()
        mock_pdf.assert_not_called()

    def test_process_pdf_to_markdown_routes_pdf_to_pdf_extraction(self):
        """PDF files must still go to _claude_extract_pdf."""
        processor = self._get_processor()

        with (
            patch.object(processor, "_claude_extract_pdf", return_value="<!-- PAGE 1 -->\nFinancials") as mock_pdf,
            patch.object(processor, "_claude_extract_image") as mock_image,
        ):
            processor._process_pdf_to_markdown(Path("/tmp/report.pdf"), "/tmp/images")

        mock_pdf.assert_called_once()
        mock_image.assert_not_called()

    def test_image_extraction_falls_back_on_error(self):
        """If Claude image extraction fails, fall back to pymupdf4llm."""
        processor = self._get_processor()

        with (
            patch.object(processor, "_claude_extract_image", side_effect=Exception("API error")),
            patch.object(processor, "_pymupdf4llm_extract", return_value="fallback content") as mock_fallback,
        ):
            result = processor._process_pdf_to_markdown(Path("/tmp/test.jpg"), "/tmp/images")

        mock_fallback.assert_called_once()
        assert result == "fallback content"

    def test_claude_extract_image_sends_correct_content_block(self):
        """_claude_extract_image must call invoke_image with correct media_type."""
        processor = self._get_processor()

        mock_result = Mock()
        mock_result.content = "<!-- PAGE 1 -->\nExtracted text"
        mock_result.input_tokens = 100
        mock_result.output_tokens = 50

        mock_client = Mock()
        mock_client.invoke_image.return_value = mock_result
        mock_client.log_model = "test-model"

        with (
            patch("scribe.tools.ocr_processor.get_llm_client", return_value=mock_client),
            patch.object(Path, "read_bytes", return_value=b"fake-image-data"),
        ):
            result = processor._claude_extract_image(Path("/tmp/photo.jpeg"))

        # Verify invoke_image was called with correct media_type
        call_args = mock_client.invoke_image.call_args
        assert call_args[0][1] == "image/jpeg"  # media_type argument
        assert result == "<!-- PAGE 1 -->\nExtracted text"

    def test_unknown_extension_skips_claude_uses_fallback(self):
        """Unknown file extensions (e.g., .tiff) should use pymupdf4llm fallback."""
        processor = self._get_processor()

        with (
            patch.object(processor, "_claude_extract_image") as mock_image,
            patch.object(processor, "_claude_extract_pdf") as mock_pdf,
            patch.object(processor, "_pymupdf4llm_extract", return_value="tiff content") as mock_fallback,
        ):
            # .tiff is not in IMAGE_MEDIA_TYPES and not .pdf
            # The strategy is "claude" but neither image nor pdf handler matches
            # _claude_extract_pdf will be called (it's the default for non-image claude strategy)
            mock_pdf.side_effect = Exception("not a PDF")
            processor._process_pdf_to_markdown(Path("/tmp/scan.tiff"), "/tmp/images")

        # Should have tried PDF extraction and fallen back
        mock_image.assert_not_called()
        mock_pdf.assert_called_once()
        mock_fallback.assert_called_once()
