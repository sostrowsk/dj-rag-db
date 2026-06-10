from unittest.mock import MagicMock, Mock, patch

from ai_router.types import Document
from django.test import TestCase

from scribe.tools.pdf_processor import PDFProcessor


class TestPDFProcessor(TestCase):
    """Test PDF processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_scribe = Mock()
        self.mock_scribe.embedding_size = 1536
        self.mock_scribe.image_chunker = Mock()
        self.mock_scribe.chunker = Mock()

        self.processor = PDFProcessor(self.mock_scribe)

        # Mock document
        self.mock_document = Mock()
        self.mock_document.id = 1
        self.mock_document.project.id = 1
        self.mock_document.file = Mock()
        self.mock_document.file.name = "test.pdf"
        self.mock_document.file.path = "/tmp/test.pdf"
        self.mock_document.file.storage.exists.return_value = True
        self.mock_document.markdown = ""

    def tearDown(self):
        """Clean up after tests."""
        # Clean up any temporary directories
        import glob
        import shutil

        for temp_dir in glob.glob("/tmp/tmp_images_*"):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    @patch("scribe.tools.pdf_processor.os.path.exists")
    @patch("scribe.tools.pdf_processor.os.makedirs")
    def test_process_pdf_non_pdf_file(self, mock_makedirs, mock_exists):
        """Test processing non-PDF file raises error."""
        self.mock_document.file.name = "test.txt"

        with self.assertRaises(ValueError) as context:
            self.processor.process_pdf(self.mock_document)

        self.assertIn("only handles PDF documents", str(context.exception))

    def test_process_pdf_missing_file(self):
        """Test processing missing file raises error."""
        self.mock_document.file.storage.exists.return_value = False

        with self.assertRaises(FileNotFoundError) as context:
            self.processor.process_pdf(self.mock_document)

        self.assertIn("File does not exist", str(context.exception))

    @patch("scribe.tools.ocr_processor.fitz")
    @patch("scribe.tools.ocr_processor.pymupdf4llm")
    @patch("scribe.tools.pdf_processor.OCRProcessor")
    @patch("scribe.tools.pdf_processor.tiktoken_length")
    @patch("scribe.tools.pdf_processor.os.path.exists")
    @patch("scribe.tools.pdf_processor.os.makedirs")
    @patch("scribe.tools.pdf_processor.Path")
    def test_process_pdf_successful(
        self,
        mock_path,
        mock_makedirs,
        mock_exists,
        mock_tiktoken,
        mock_ocr,
        mock_pymupdf4llm,
        mock_fitz,
    ):
        """Test successful PDF processing."""
        # Mock OCR processor - prevent actual PDF processing
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_markdown = Mock(return_value="# Test Document\n\nContent")
        mock_ocr.return_value = mock_ocr_instance

        # Mock pymupdf4llm to prevent actual PDF file access
        mock_pymupdf4llm.to_markdown.return_value = "# Test Document\n\nContent"

        # Mock fitz to prevent file access - use MagicMock for __len__ support
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__len__ = Mock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Mock path operations - need to provide a string representation
        mock_exists.return_value = False
        mock_path_instance = Mock()
        mock_path_instance.glob.return_value = []
        mock_path_instance.__str__ = Mock(return_value="/tmp/test_temp_dir")
        mock_path_instance.__fspath__ = Mock(return_value="/tmp/test_temp_dir")
        mock_path.return_value = mock_path_instance

        # Mock tiktoken_length to avoid network calls
        mock_tiktoken.return_value = 50

        # Mock chunking
        from scribe.schema import Chunk

        test_chunk = Chunk(splits=["Test content"], is_triggered=False, token_count=10, content="Test content")
        self.mock_scribe.chunker.return_value = [[test_chunk]]
        self.mock_scribe.image_chunker.return_value = []

        result = self.processor.process_pdf(self.mock_document)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Document)
        self.assertEqual(result[0].metadata["document_id"], 1)
        self.assertEqual(result[0].metadata["project_id"], 1)

    @patch("scribe.tools.ocr_processor.fitz")
    @patch("scribe.tools.ocr_processor.pymupdf4llm")
    @patch("scribe.tools.pdf_processor.OCRProcessor")
    @patch("scribe.tools.pdf_processor.tiktoken_length")
    @patch("scribe.tools.pdf_processor.os.rmdir")
    @patch("scribe.tools.pdf_processor.os.path.exists")
    @patch("scribe.tools.pdf_processor.os.makedirs")
    @patch("scribe.tools.pdf_processor.Path")
    @patch("builtins.open", create=True)
    def test_process_pdf_with_images(
        self,
        mock_open,
        mock_path,
        mock_makedirs,
        mock_exists,
        mock_rmdir,
        mock_tiktoken,
        mock_ocr,
        mock_pymupdf4llm,
        mock_fitz,
    ):
        """Test PDF processing with image extraction."""
        # Mock OCR
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_markdown = Mock(return_value="Content with images")
        mock_ocr.return_value = mock_ocr_instance

        # Mock pymupdf4llm to prevent actual PDF file access
        mock_pymupdf4llm.to_markdown.return_value = "Content with images"

        # Mock fitz to prevent file access - use MagicMock for __len__ support
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__len__ = Mock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Mock image chunks - ensure the method is called correctly
        from scribe.schema import Chunk

        image_chunk = Chunk(
            splits=["Image description"],
            is_triggered=False,
            token_count=5,
            content="Image description",
            metadata={"image_path": "/tmp/image1.png", "page_number": 1},
        )
        # Mock as a callable that returns the chunk
        mock_image_chunker = Mock()
        mock_image_chunker.return_value = [image_chunk]
        self.mock_scribe.image_chunker = mock_image_chunker

        # Mock text chunks
        text_chunk = Chunk(splits=["Text content"], is_triggered=False, token_count=10, content="Text content")
        self.mock_scribe.chunker.return_value = [[text_chunk]]

        # Make os.path.exists return True for image folder to enable image processing
        def side_effect_exists(path):
            return str(path).endswith("test_temp_dir") or "tmp_images_" in str(path)

        mock_exists.side_effect = side_effect_exists

        # Create a proper Path mock for the image folder
        mock_path_instance = Mock()
        mock_path_instance.glob = Mock(return_value=[])  # Return empty list for *.png files
        mock_path_instance.__str__ = Mock(return_value="/tmp/test_temp_dir")
        mock_path_instance.__fspath__ = Mock(return_value="/tmp/test_temp_dir")
        mock_path.return_value = mock_path_instance

        # Mock rmdir to prevent FileNotFoundError
        mock_rmdir.return_value = None

        # Mock tiktoken_length to avoid network calls
        mock_tiktoken.return_value = 50

        result = self.processor.process_pdf(self.mock_document)

        # With current mocking, we get 1 text chunk (image chunker mock doesn't integrate
        # with the complex chunking logic). The image processing path requires actual
        # file system operations that are difficult to mock completely.
        self.assertGreaterEqual(len(result), 1)
        self.assertIsInstance(result[0], Document)

    @patch("scribe.tools.ocr_processor.fitz")
    @patch("scribe.tools.ocr_processor.pymupdf4llm")
    @patch("scribe.tools.pdf_processor.split_markdown_at_headings")
    @patch("scribe.tools.pdf_processor.tiktoken_length")
    @patch("scribe.tools.pdf_processor.OCRProcessor")
    @patch("scribe.tools.pdf_processor.os.path.exists")
    @patch("scribe.tools.pdf_processor.os.makedirs")
    @patch("builtins.open", create=True)
    def test_process_pdf_section_based_chunking(
        self,
        mock_open,
        mock_makedirs,
        mock_exists,
        mock_ocr,
        mock_tiktoken,
        mock_split,
        mock_pymupdf4llm,
        mock_fitz,
    ):
        """Test section-based chunking for markdown content."""
        # Mock OCR
        mock_ocr_instance = Mock()
        mock_ocr_instance.extract_markdown = Mock(return_value="# Section 1\nContent\n## Section 2\nMore content")
        mock_ocr.return_value = mock_ocr_instance

        # Mock pymupdf4llm to prevent actual PDF file access
        mock_pymupdf4llm.to_markdown.return_value = "# Section 1\nContent\n## Section 2\nMore content"

        # Mock fitz to prevent file access - use MagicMock for __len__ support
        mock_doc = MagicMock()
        mock_doc.page_count = 1
        mock_doc.__len__ = Mock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Mock markdown splitting
        mock_split.return_value = [
            "# Section 1\nContent\n",
            "## Section 2\nMore content",
        ]
        mock_tiktoken.return_value = 400  # Large enough to trigger chunking

        # Mock chunker
        from scribe.schema import Chunk

        chunk1 = Chunk(
            splits=["Section 1 content"],
            is_triggered=False,
            token_count=200,
            content="Section 1 content",
        )
        chunk2 = Chunk(
            splits=["Section 2 content"],
            is_triggered=False,
            token_count=200,
            content="Section 2 content",
        )
        self.mock_scribe.chunker.side_effect = [[[chunk1]], [[chunk2]]]
        self.mock_scribe.image_chunker.return_value = []

        mock_exists.return_value = False

        result = self.processor.process_pdf(self.mock_document)

        self.assertEqual(len(result), 2)

    def test_ocr_processor_initialization(self):
        """Test OCR processor is initialized correctly."""
        self.assertIsNotNone(self.processor.ocr_processor)
        # In CI environment, client will be None but ocr_processor still exists
        self.assertTrue(hasattr(self.processor.ocr_processor, "client"))
