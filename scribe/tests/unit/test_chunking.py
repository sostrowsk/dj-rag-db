from unittest import TestCase
from unittest.mock import Mock, patch

from scribe.chunking.imagechunker import ImageChunker
from scribe.chunking.propositionchunker import PropositionChunker
from scribe.chunking.statisticalchunker import StatisticalChunker_GaussianSmoothing
from scribe.schema import Chunk


class TestStatisticalChunker(TestCase):
    """Test statistical chunking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a proper encoder mock that inherits from BaseEncoder
        from ai_router.encoders import BaseEncoder

        class MockEncoder(BaseEncoder):
            def __init__(self):
                super().__init__(name="test_encoder")
                self._encode_return_value = [5]  # Default to 5 tokens

            def __call__(self, texts):
                return [[0.1] * 10 for _ in texts]

            def encode(self, texts):
                # Return the configured token counts
                if hasattr(self, "_encode_return_value"):
                    if isinstance(self._encode_return_value, list):
                        return self._encode_return_value[: len(texts)]
                    return [self._encode_return_value for _ in texts]
                return [5 for _ in texts]

            class Config:
                extra = "allow"

        self.mock_encoder = MockEncoder()

        # Create a proper splitter mock
        from scribe.splitters import BaseSplitter

        class MockSplitter(BaseSplitter):
            def __call__(self, text):
                # BaseSplitter expects __call__ method
                return self.split_text(text)

            def split_text(self, text):
                if not text:
                    return []
                return ["This is a test.", "Another sentence.", "Final sentence."]

        self.mock_splitter = MockSplitter()

        self.chunker = StatisticalChunker_GaussianSmoothing(
            encoder=self.mock_encoder,
            splitter=self.mock_splitter,
            name="test_chunker",
            min_split_tokens=5,
            max_split_tokens=100,
            dynamic_threshold=True,
            window_size=3,
        )

    def test_chunker_initialization(self):
        """Test chunker initializes with correct parameters."""
        self.assertEqual(self.chunker.name, "test_chunker")
        self.assertEqual(self.chunker.min_split_tokens, 5)
        self.assertEqual(self.chunker.max_split_tokens, 100)
        self.assertTrue(self.chunker.dynamic_threshold)

    def test_chunking_empty_text(self):
        """Test chunking with empty text."""
        result = self.chunker([""])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 0)

    def test_chunking_small_text(self):
        """Test chunking with text smaller than min tokens."""
        self.mock_encoder._encode_return_value = [3]  # 3 tokens < 5 min
        result = self.chunker(["Small text"])
        self.assertIsInstance(result, list)


@patch("scribe.chunking.basechunker.get_llm_client")
class TestPropositionChunker(TestCase):
    """Test proposition-based chunking."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_proposition_chunker_init(self, mock_client):
        """Test proposition chunker initialization."""
        mock_llm = Mock()
        mock_client.return_value = mock_llm
        chunker = PropositionChunker(model="gpt-5.4")
        self.assertIsNotNone(chunker.client)

    @patch("scribe.chunking.propositionchunker.PropositionChunker.get_propositions_from_text")
    def test_proposition_extraction(self, mock_get_propositions, mock_client):
        """Test extracting propositions from text."""
        mock_llm = Mock()
        mock_client.return_value = mock_llm

        # Mock get_propositions_from_text to return a list of propositions
        mock_get_propositions.return_value = ["Proposition 1", "Proposition 2"]

        # Mock response for invoke() calls (returns tuple of (result, usage))
        mock_heading_response = Mock()
        mock_heading_response.content = "Test Heading"
        mock_heading_response.input_tokens = 10
        mock_heading_response.output_tokens = 5
        mock_llm.invoke.return_value = (mock_heading_response, None)
        mock_llm.log_model = "test-model"

        chunker = PropositionChunker(model="gpt-5.4")
        text = "This is a complex paragraph with multiple ideas."
        result = chunker([], text)

        mock_get_propositions.assert_called_once()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)  # One chunk should be created
        self.assertIn("Proposition 1", result[0].splits[0])
        self.assertIn("Proposition 2", result[0].splits[0])


class TestImageChunker(TestCase):
    """Test image chunking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        with patch("scribe.chunking.basechunker.get_llm_client") as mock_client:
            self.mock_llm = Mock()
            mock_client.return_value = self.mock_llm
            self.chunker = ImageChunker(model="gpt-5.4")

    def test_image_chunker_empty_folder(self):
        """Test chunking with empty image folder."""
        from pathlib import Path

        mock_path = Mock(spec=Path)
        mock_path.glob.return_value = []

        result = self.chunker(mock_path)
        self.assertEqual(len(result), 0)

    def test_image_chunker_with_images(self):
        """Test chunking with image files."""
        from pathlib import Path

        # Mock image files
        mock_image1 = Mock(spec=Path)
        mock_image1.name = "image1.png"

        # Create a proper context manager mock for file opening
        mock_file = Mock()
        mock_file.read.return_value = b"fake_image_data"
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_file)
        mock_context.__exit__ = Mock(return_value=None)
        mock_image1.open.return_value = mock_context

        mock_path = Mock(spec=Path)
        mock_path.glob.return_value = [mock_image1]

        # Mock invoke() to return (result, parsed) tuple
        mock_result = Mock()
        mock_result.content = "Description of image"
        mock_result.input_tokens = 10
        mock_result.output_tokens = 5
        mock_parsed = Mock()
        mock_parsed.propositions = ["Description of image"]
        self.mock_llm.invoke.return_value = (mock_result, mock_parsed)
        self.mock_llm.log_model = "test-model"

        result = self.chunker(mock_path)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Chunk)
