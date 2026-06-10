"""Regression tests for: Image-Chunks kollidieren mit Text-Chunk-IDs (Codex P2).

ImageChunker vergibt in der Chunk-Metadata ein eigenes ``idx`` ab 0, waehrend
Text-Chunks ueber den globalen Listen-Index nummeriert werden. Unter dem
UniqueConstraint (collection_name, document_id, chunk_id) wurden Image-Chunks
beim ``bulk_create(ignore_conflicts=True)`` dadurch still verworfen —
Bild-Inhalte waren nicht durchsuchbar, der Insert meldete trotzdem Erfolg.
"""

from unittest.mock import Mock

from django.test import SimpleTestCase

from scribe.schema import Chunk
from scribe.tools.pdf_processor import build_chunk_documents


def _document_mock():
    document = Mock()
    document.id = 42
    document.project = Mock(id=7)
    document.file = None
    return document


def _text_chunk(content):
    return Chunk(splits=[content], is_triggered=False, token_count=10, content=content)


def _image_chunk(content, idx):
    return Chunk(
        splits=[content],
        is_triggered=False,
        token_count=5,
        content=content,
        metadata={"idx": idx, "image_path": f"/tmp/img_{idx}.png", "page_number": 1},
    )


class TestChunkIdUniqueness(SimpleTestCase):
    def test_text_and_image_chunks_get_unique_chunk_ids(self):
        """Text-Chunk #0 und Image-Chunk mit metadata idx=0 duerfen nicht kollidieren."""
        chunks = [_text_chunk("Text A"), _text_chunk("Text B"), _image_chunk("Bild 1", idx=0)]
        valid_chunks = list(enumerate(chunks))

        docs = build_chunk_documents(valid_chunks, _document_mock())

        chunk_ids = [doc.metadata["chunk_id"] for doc in docs]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), f"chunk_ids not unique: {chunk_ids}")

    def test_chunk_ids_are_sequential_list_positions(self):
        """chunk_id entspricht der globalen Listen-Position (stabile Reihenfolge)."""
        chunks = [_text_chunk("Text A"), _image_chunk("Bild 1", idx=0), _image_chunk("Bild 2", idx=1)]
        valid_chunks = list(enumerate(chunks))

        docs = build_chunk_documents(valid_chunks, _document_mock())

        self.assertEqual([doc.metadata["chunk_id"] for doc in docs], [0, 1, 2])

    def test_image_metadata_is_preserved(self):
        """image_path/page_number aus der Chunk-Metadata bleiben erhalten."""
        chunks = [_text_chunk("Text A"), _image_chunk("Bild 1", idx=0)]
        docs = build_chunk_documents(list(enumerate(chunks)), _document_mock())

        self.assertEqual(docs[1].metadata["image_path"], "/tmp/img_0.png")
        self.assertEqual(docs[1].metadata["page_number"], 1)
