from unittest.mock import Mock, patch

from django.test import TestCase

from scribe.processing.contextualizer import DocumentContextualizer


class TestDocumentContextualizer(TestCase):
    """Test document contextualization."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()

        with patch("scribe.processing.contextualizer.get_llm_client") as mock_get_client:
            mock_get_client.return_value = self.mock_client
            self.contextualizer = DocumentContextualizer(model="test-model")

    def test_contextualizer_init(self):
        """Test contextualizer initialization."""
        self.assertIsNotNone(self.contextualizer.client)

    def test_contextualize_chunk(self):
        """Test contextualizing a single chunk."""
        from ai_router.types import Document

        chunk = Document(
            page_content="This is a chunk about machine learning.",
            metadata={"chunk_id": 1},
        )
        whole_document = "This document discusses AI and machine learning..."

        mock_response = Mock()
        mock_response.content = "Context: This chunk discusses ML concepts from the AI document."
        mock_response.input_tokens = 100
        mock_response.output_tokens = 20
        self.mock_client.invoke.return_value = (mock_response, None)

        result = self.contextualizer.contextualize_chunk(chunk, whole_document)

        self.assertIsInstance(result, Document)
        self.assertTrue(result.metadata["has_context"])
        self.assertIn("<context>", result.page_content)

    def test_contextualize_chunk_error_handling(self):
        """Test error handling during contextualization."""
        from ai_router.types import Document

        chunk = Document(page_content="Test chunk")

        self.mock_client.invoke.side_effect = Exception("LLM error")

        result = self.contextualizer.contextualize_chunk(chunk, "Document text")

        # Should return original chunk on error
        self.assertEqual(result.page_content, chunk.page_content)
        self.assertFalse(result.metadata.get("has_context", False))

    def test_contextualize_large_document(self):
        """Test handling of large documents."""
        from ai_router.types import Document

        chunk = Document(page_content="Test chunk")
        large_document = "x" * 150000

        mock_response = Mock()
        mock_response.content = "Context"
        mock_response.input_tokens = 100
        mock_response.output_tokens = 10
        self.mock_client.invoke.return_value = (mock_response, None)

        result = self.contextualizer.contextualize_chunk(chunk, large_document)

        # Check that document was truncated (100k chars) in the user_prompt
        call_args = self.mock_client.invoke.call_args[0]
        user_prompt = call_args[1]
        # The truncated document (100k chars) should be in the prompt
        self.assertNotIn("x" * 150000, user_prompt)
        self.assertIn("<context>", result.page_content)
