# scribe/tools/document_to_pdf.py
import logging
import subprocess
import tempfile
from pathlib import Path

import pypandoc

logger = logging.getLogger(__name__)


def document_to_pdf(input_path, output_path, input_format):
    extra_args = ["--pdf-engine=xelatex"]
    pypandoc.convert_file(
        str(input_path),
        "pdf",
        outputfile=str(output_path),
        format=input_format,
        extra_args=extra_args,
    )
    compress_pdf(output_path)


def compress_pdf(pdf_path):
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        subprocess.run(
            [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/ebook",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                f"-sOutputFile={temp_path}",
                str(pdf_path),
            ],
            check=True,
        )
        original_size = Path(pdf_path).stat().st_size
        compressed_size = temp_path.stat().st_size
        if compressed_size < original_size:
            temp_path.rename(pdf_path)
            reduction_percent = (1 - compressed_size / original_size) * 100
            logger.info(
                f"PDF compressed from {original_size/1024:.1f}KB to {compressed_size/1024:.1f}KB "
                f"({reduction_percent:.1f}% reduction)"
            )
        else:
            temp_path.unlink()
            logger.info("Compression did not reduce file size, keeping original")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Ghostscript compression failed: {e}. Keeping original PDF.")
        if temp_path.exists():
            temp_path.unlink()
    except FileNotFoundError:
        logger.warning("Ghostscript not installed. PDF compression skipped.")
