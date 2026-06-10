# scribe/tools/ocr_processor.py
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz
import pymupdf4llm
from ai_router.client import get_llm_client
from django.conf import settings

logger = logging.getLogger(__name__)

# Page separator for markdown extraction - inserted between PDF pages
PAGE_SEPARATOR = "<!-- PAGE {page_num} -->"


def _has_broken_table(page_md: str) -> bool:
    """Detect broken tables where pymupdf4llm collapsed columns into <br>-separated cells."""
    for line in page_md.splitlines():
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        for cell in cells:
            if cell.count("<br>") > 5:
                return True
    return False


def _count_data_columns(rows: list[list[str]]) -> int:
    """Count numeric data columns by scanning right-to-left.

    Args:
        rows: list of lists of strings from table.extract().
    """
    if not rows or not rows[0]:
        return 8
    total_cols = len(rows[0])
    numeric_pattern = re.compile(r"^-?\d[\d.,]*%?$")
    data_cols = 0
    for col_idx in range(total_cols - 1, -1, -1):
        non_empty = 0
        numeric = 0
        for row in rows:
            val = (row[col_idx] or "").strip() if col_idx < len(row) else ""
            if val:
                non_empty += 1
                if numeric_pattern.match(val):
                    numeric += 1
        if non_empty > 0 and numeric / non_empty > 0.4:
            data_cols += 1
        else:
            break
    return data_cols if data_cols > 0 else 8


def _build_markdown_table(rows: list[list[str]], data_cols: int) -> str:
    """Build a clean markdown table merging fragmented label columns.

    Args:
        rows: list of lists of strings from table.extract().
        data_cols: number of right-hand numeric data columns.
    """
    if not rows:
        return ""
    total_cols = len(rows[0])
    label_cols = total_cols - data_cols

    md_lines = []
    for i, row in enumerate(rows):
        cells = [c or "" for c in row]
        # Pad if row has fewer cells
        while len(cells) < total_cols:
            cells.append("")
        # Merge label columns
        label_parts = [c.strip() for c in cells[:label_cols] if c.strip()]
        label = " ".join(label_parts)
        data = [c.strip() for c in cells[label_cols:]]
        # Skip fully empty rows
        if not label and not any(data):
            continue
        md_lines.append("| " + " | ".join([label] + data) + " |")
        # Insert header separator after first row
        if i == 0:
            md_lines.append("| " + " | ".join(["---"] * (1 + data_cols)) + " |")

    return "\n".join(md_lines)


def _replace_broken_table(page_md: str, page: fitz.Page) -> str:
    """Replace broken table block with rebuilt table from find_tables(strategy='text')."""
    tables = page.find_tables(strategy="text")
    if not tables.tables:
        return ""

    # Use first (largest) table — extract() returns list[list[str]]
    table = tables[0]
    rows = table.extract()
    if not rows:
        return ""

    data_cols = _count_data_columns(rows)
    rebuilt = _build_markdown_table(rows, data_cols)
    if not rebuilt:
        return ""

    # Find the contiguous block of table lines in page_md and replace it
    lines = page_md.splitlines()
    table_start = None
    table_end = None
    for idx, line in enumerate(lines):
        if line.startswith("|"):
            if table_start is None:
                table_start = idx
            table_end = idx
        elif table_start is not None:
            break

    if table_start is None:
        return ""

    before = "\n".join(lines[:table_start])
    after = "\n".join(lines[table_end + 1 :])
    parts = [p for p in [before.rstrip(), rebuilt, after.lstrip()] if p]
    return "\n\n".join(parts)


@dataclass
class OCRSettings:
    tessdata_dir: str = getattr(settings, "TESSDATA_DIR", str(Path(__file__).parent.parent / "tessdata"))
    tesseract_cmd: str = getattr(settings, "TESSERACT_CMD", "tesseract")
    tesseract_timeout: int = getattr(settings, "TESSERACT_TIMEOUT", 300)
    language: str = getattr(settings, "TESSERACT_LANGUAGE", "eng+deu")
    dpi: int = getattr(settings, "TESSERACT_DPI", 300)


class LLMOutputError(Exception):
    pass


class OCRProcessor:
    def __init__(self, ocr_settings: Optional[OCRSettings] = None, progress=None):
        self.ocr_settings = ocr_settings or OCRSettings()
        try:
            self.client = get_llm_client(settings.DEFAULT_MODEL_SCRIBE_OCR_PROCESSOR)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")
            self.client = None
        self.progress = progress
        if self.ocr_settings.tessdata_dir:
            tessdata_path = Path(self.ocr_settings.tessdata_dir)
            if not tessdata_path.exists():
                logger.error(f"TESSDATA_PREFIX path does not exist: {tessdata_path}")
            elif not list(tessdata_path.glob("*.traineddata")):
                logger.error(f"No Tesseract language data found in {tessdata_path}")

    def _vision_ocr_page(self, page: fitz.Page) -> str:
        """Use Claude Vision via Bedrock to extract text from a scanned page."""
        import base64

        from ai_router.logging import llm_log

        pix = page.get_pixmap(dpi=self.ocr_settings.dpi)
        png_bytes = pix.tobytes("png")
        base64_png = base64.b64encode(png_bytes).decode("utf-8")

        client = get_llm_client()

        user_text = (
            "Extract ALL text from this scanned document page as structured markdown. "
            "Preserve table structures using markdown table syntax. "
            "Keep all numbers, currencies and special characters exactly as shown. "
            "Output ONLY the extracted text, no commentary."
        )

        with llm_log("ocr_vision_page", client.log_model, user_prompt=user_text) as log:
            result = client.invoke_image(base64_png, "image/png", user_text)
            log.output = result.content
            log.input_tokens = result.input_tokens
            log.output_tokens = result.output_tokens
        return result.content

    def _claude_extract_pdf(self, pdf_path: Path) -> str:
        """Send entire PDF to Claude via CachedClient for markdown extraction.

        Returns extracted markdown with <!-- PAGE N --> markers, or empty string on failure.
        """
        from ai_router.logging import llm_log

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        client = get_llm_client()

        prompt = (
            "Extract ALL text content from this PDF document as clean, structured markdown.\n\n"
            "CRITICAL RULES:\n"
            f"1. PAGE MARKERS: This document has {total_pages} pages. "
            f'Insert "<!-- PAGE N -->" at the beginning of each page\'s content '
            f"(N = 1 to {total_pages}). Every page must have its marker.\n"
            "2. FINANCIAL TABLES: Extract ALL numbers exactly as shown. "
            "Do not skip any numeric values from balance sheets, income statements, "
            "cash flow statements, or notes.\n"
            "3. TABLE FORMAT: Use markdown pipe-table syntax:\n"
            "   | Label | 2022 | % | 2021 | % |\n"
            "   | --- | --- | --- | --- | --- |\n"
            "   | Row | 1.234 | 38 | 5.678 | 37 |\n"
            "4. PRESERVE: All numbers, currencies, percentages, dates exactly as they appear.\n"
            "5. STRUCTURE: Use # headings, bold, lists as in the original.\n"
            "6. OUTPUT: Return ONLY the extracted markdown. No commentary, no wrapping tags."
        )

        max_tokens = getattr(settings, "PDF_EXTRACTION_MAX_TOKENS", 32000)
        logger.info(f"Sending {total_pages}-page PDF to Claude for extraction: {pdf_path.name}")

        with llm_log("ocr_extract_pdf", client.log_model, user_prompt=prompt) as log:
            result = client.stream_pdf(str(pdf_path), prompt, max_tokens=max_tokens)
            log.output = result.content
            log.input_tokens = result.input_tokens
            log.output_tokens = result.output_tokens

        # Validate page markers
        page_markers = re.findall(r"<!-- PAGE (\d+) -->", result.content)
        if len(page_markers) < total_pages * 0.5:
            logger.warning(
                f"Claude extraction has only {len(page_markers)}/{total_pages} page markers, " "result may be malformed"
            )
            return ""

        logger.info(
            f"Claude PDF extraction: {len(result.content)} chars, " f"{len(page_markers)}/{total_pages} page markers"
        )
        return result.content

    # extract_all_from_pdf removed — GuV/Bilanz extraction now uses separate PDF-based tasks

    IMAGE_MEDIA_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }

    def _claude_extract_image(self, image_path: Path) -> str:
        """Send an image file to Claude via Bedrock for markdown extraction."""
        import base64

        suffix = image_path.suffix.lower()
        media_type = self.IMAGE_MEDIA_TYPES.get(suffix)
        if not media_type:
            return ""

        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        client = get_llm_client()

        prompt = (
            "Extract ALL text content from this image as clean, structured markdown.\n\n"
            "RULES:\n"
            '1. Insert "<!-- PAGE 1 -->" at the beginning of the content.\n'
            "2. Preserve all numbers, tables, and formatting exactly as shown.\n"
            "3. Use markdown pipe-table syntax for any tables.\n"
            "4. Return ONLY the extracted markdown. No commentary."
        )

        from ai_router.logging import llm_log

        logger.info(f"Sending image to Claude for extraction: {image_path.name}")

        with llm_log("ocr_extract_image", client.log_model, user_prompt=prompt) as log:
            result = client.invoke_image(image_data, media_type, prompt, max_tokens=32000)
            log.output = result.content
            log.input_tokens = result.input_tokens
            log.output_tokens = result.output_tokens
        logger.info(f"Claude image extraction: {len(result.content)} chars")
        return result.content

    def _tesseract_ocr_page(self, page: fitz.Page) -> str:
        """Fallback: use PyMuPDF's built-in Tesseract OCR."""
        text_page = page.get_textpage_ocr(
            language=self.ocr_settings.language,
            dpi=self.ocr_settings.dpi,
            full=True,
        )
        return page.get_text(textpage=text_page)

    def process_page(self, page: fitz.Page) -> str:
        try:
            text = page.get_text()
            if text.strip():
                logger.debug(f"Direct text extraction: {text[:100]}...")
                return text

            if not page.get_images():
                logger.warning(f"No text or images found on page {page.number + 1}")
                return ""

            logger.info(f"Performing OCR on page {page.number + 1}")

            # Try Claude Vision first
            vision_enabled = getattr(settings, "VISION_OCR_ENABLED", True)
            if vision_enabled:
                try:
                    result = self._vision_ocr_page(page)
                    if result and result.strip():
                        logger.info(f"Vision OCR extracted {len(result)} chars from page {page.number + 1}")
                        return result
                except Exception as e:
                    logger.warning(f"Vision OCR failed for page {page.number + 1}: {e}")

            # Fallback to Tesseract
            try:
                ocr_text = self._tesseract_ocr_page(page)
                if ocr_text and ocr_text.strip():
                    logger.info(f"Tesseract OCR extracted {len(ocr_text)} chars from page {page.number + 1}")
                    return ocr_text
            except Exception as e:
                logger.warning(f"Tesseract OCR failed for page {page.number + 1}: {e}")

            logger.warning(f"All OCR methods failed for page {page.number + 1}")
            return ""
        except Exception as e:
            logger.error(f"Error processing page {page.number + 1}: {str(e)}")
            return ""

    def paresed_result(self, input_text, tag):
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        start_index = input_text.find(start_tag)
        if start_index == -1:
            raise LLMOutputError(f"Start tag '{start_tag}' not found in LLM response.")
        start_pos = start_index + len(start_tag)
        end_index = input_text.find(end_tag, start_pos)
        if end_index == -1:
            raise LLMOutputError(f"End tag '{end_tag}' not found in LLM response.")
        extracted_text = input_text[start_pos:end_index]
        return extracted_text

    def extract_markdown(self, pdf_path: str | Path, image_folder: str | Path, save_md: bool = False) -> str:
        pdf_path = Path(pdf_path)
        output_file = pdf_path.with_suffix(".md")
        if save_md and output_file.exists():
            logger.info(f"Markdown file already exists: {output_file}")
            if self.progress:
                self.progress.log_warning(f"Using existing Markdown: {output_file}")
            return output_file.read_text(encoding="utf-8")
        markdown_output = self._process_pdf_to_markdown(pdf_path, image_folder)
        if save_md:
            try:
                output_file.write_text(markdown_output, encoding="utf-8")
                logger.info(f"Markdown saved to: {output_file}")
                if self.progress:
                    self.progress.log_warning(f"Markdown saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save Markdown to {output_file}: {str(e)}")
                if self.progress:
                    self.progress.log_error(f"Failed to save Markdown: {str(e)}")
        return markdown_output

    def _process_pdf_to_markdown(self, pdf_path: Path, image_folder: str | Path) -> str:
        logger.info(f"Processing PDF: {pdf_path}")
        if self.progress:
            self.progress.increment()
            self.progress.log_warning(f"Starting processing for: {pdf_path.name}")

        strategy = getattr(settings, "PDF_EXTRACTION_STRATEGY", "claude")
        suffix = pdf_path.suffix.lower()
        is_image = suffix in self.IMAGE_MEDIA_TYPES

        if strategy == "claude":
            try:
                if is_image:
                    result = self._claude_extract_image(pdf_path)
                else:
                    result = self._claude_extract_pdf(pdf_path)
                if result and result.strip():
                    logger.info(f"Claude extraction succeeded ({len(result)} chars)")
                    if self.progress:
                        self.progress.increment()
                        self.progress.log_warning("Content extracted with Claude")
                    return result
                logger.warning("Claude extraction returned empty, falling back to pymupdf4llm")
            except Exception as e:
                logger.warning(f"Claude extraction failed: {e}, falling back to pymupdf4llm")

        return self._pymupdf4llm_extract(pdf_path, image_folder)

    def _pymupdf4llm_extract(self, pdf_path: Path, image_folder: str | Path) -> str:
        """Extract markdown using pymupdf4llm with OCR fallback."""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Extract markdown page by page to insert page markers
        markdown_pages = []
        for page_num in range(total_pages):
            logger.debug(f"Extracting page {page_num + 1} of {total_pages}")

            if image_folder:
                page_md = pymupdf4llm.to_markdown(
                    doc=pdf_path,
                    pages=[page_num],
                    write_images=True,
                    image_path=image_folder,
                    table_strategy="lines",
                )
            else:
                page_md = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])

            # Remove image links
            page_md_clean = re.sub(r"!\[\]\(.*?\)", "", page_md)

            # Repair broken tables using text-based column detection
            if _has_broken_table(page_md_clean):
                rebuilt = _replace_broken_table(page_md_clean, doc[page_num])
                if rebuilt:
                    page_md_clean = rebuilt
                    logger.info(f"Page {page_num + 1}: rebuilt broken table via find_tables(strategy='text')")

            # Add page marker
            page_marker = f"\n\n{PAGE_SEPARATOR.format(page_num=page_num + 1)}\n\n"
            markdown_pages.append(page_marker + page_md_clean)

        # Combine all pages
        ocr_md = "\n".join(markdown_pages)

        # Check if actual content was extracted (strip page markers + whitespace)
        content_only = re.sub(r"<!--\s*PAGE\s*\d+\s*-->", "", ocr_md).strip()
        if len(content_only) > 10 * total_pages:
            logger.info(f"Using pymupdf4llm to extract content ({total_pages} pages with page markers)")
            if self.progress:
                self.progress.increment()
                self.progress.log_warning("PDF content extracted with pymupdf4llm")
            doc.close()
            return ocr_md

        # Fallback: use page-by-page OCR processing with page markers
        logger.info("Falling back to page-by-page OCR processing")
        if self.progress:
            self.progress.log_warning("Falling back to page-by-page OCR processing")
        markdown_pages = []
        try:
            for page_number, page in enumerate(doc):
                logger.debug(f"Processing page {page_number + 1} of {total_pages}")
                if self.progress:
                    self.progress.log_warning(f"OCR processing page {page_number + 1} of {total_pages}")
                try:
                    page_text = self.process_page(page)
                    if page_text.strip():
                        # Add page marker
                        page_marker = f"\n\n{PAGE_SEPARATOR.format(page_num=page_number + 1)}\n\n"
                        markdown_pages.append(page_marker + page_text)
                        logger.debug(f"Page {page_number + 1} content:\n{page_text[:200]}...")
                    else:
                        logger.warning(f"No content extracted from page {page_number + 1}")
                except Exception as page_error:
                    logger.error(f"Error processing page {page_number + 1}: {str(page_error)}")
                    if self.progress:
                        self.progress.log_error(f"Error processing page {page_number + 1}: {str(page_error)}")
                    continue
                if self.progress and total_pages > 1:
                    self.progress.increment()
        finally:
            doc.close()
        if not markdown_pages:
            error_msg = "No content extracted from any pages"
            logger.error(error_msg)
            if self.progress:
                self.progress.log_error(error_msg)
            return ""
        if self.progress:
            self.progress.increment()
            self.progress.log_warning("OCR processing completed, preparing final markdown")

        system_prompt = """You are a highly skilled AI trained to refine and correct text extracted from PDF documents,
outputting clean and well-formatted Markdown.

**Input:** You will receive raw text extracted from a PDF, potentially containing OCR errors, formatting issues, a
nd inconsistencies.

**Output:**
- Generate Markdown that accurately represents the original PDF's content and structure.
- Correct spelling, grammar, and punctuation errors.
- Preserve the logical flow and organization of the original document.
- Use appropriate Markdown syntax for headings, lists, tables, and other elements.
- Ensure the output is within <output></output> tags.
- If the input contains a table, preserve it using markdown table syntax.
- Example of input text:
'''
This is a sampl txt.
It has speling erors and formating isues.
1. Item one
2. Iem two
'''
- Example of output:
<output>
This is a sample text.
It has spelling errors and formatting issues.
1. Item one
2. Item two
</output>
"""

        output = []
        if self.client is None:
            logger.warning("LLM not available, returning raw markdown pages")
            for page_number, markdown_page in enumerate(markdown_pages):
                if markdown_page.strip():
                    output.append(f"<page {page_number}>\n{markdown_page}\n</page {page_number}>\n")
            return "\n".join(output) if output else ""

        for page_number, markdown_page in enumerate(markdown_pages):
            if not markdown_page.strip():
                logger.warning(f"Skipping empty page {page_number}")
                continue
            try:
                user_prompt = f"InputText: {markdown_page}"
                result, _ = self.client.invoke(system_prompt, user_prompt)
                parsed_output = self.paresed_result(result.content, "output")
                if parsed_output.strip():
                    output.append(f"<page {page_number}>\n{parsed_output}\n</page {page_number}>\n")
                else:
                    logger.warning(f"Empty LLM output for page {page_number}")
            except Exception as llm_error:
                logger.error(f"LLM processing failed for page {page_number}: {str(llm_error)}")
                output.append(f"<page {page_number}>\n{markdown_page}\n</page {page_number}>\n")
        return "\n".join(output) if output else ""


def extract_markdown_with_ocr(pdf_path: str | Path, image_folder=None, progress=None, save_md=False) -> str:
    ocr_settings = OCRSettings(tessdata_dir=settings.TESSDATA_DIR)
    processor = OCRProcessor(ocr_settings, progress=progress)
    return processor.extract_markdown(pdf_path, image_folder, save_md=save_md)
