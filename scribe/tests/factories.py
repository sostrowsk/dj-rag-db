import math
import random

import factory
from ai_router.types import Document
from factory.django import DjangoModelFactory

from scribe.models import DocumentChunk
from scribe.schema import Chunk

EMBEDDING_DIMENSIONS = 3072


def deterministic_embedding(seed: int, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Seedable, unit-normalized fake embedding.

    Same seed -> identical vector, different seeds -> different vectors.
    Unit norm makes cosine-distance ordering in tests deterministic and
    directly comparable (matches text-embedding-3-large, which is also
    unit-normalized).
    """
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dimensions)]
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class DocumentChunkFactory(DjangoModelFactory):
    """Factory for DocumentChunk ORM rows with deterministic embeddings."""

    class Meta:
        model = DocumentChunk

    collection_name = "project_1"
    project_document = None
    client_document = None
    document_id = factory.Sequence(lambda n: n + 1)
    project_id = 1
    chunk_id = factory.Sequence(lambda n: n)
    content = factory.Faker("paragraph", nb_sentences=5)
    embedding = factory.Sequence(lambda n: deterministic_embedding(seed=n))


class DocumentFactory(factory.Factory):
    """Factory for creating Langchain Document objects."""

    class Meta:
        model = Document

    page_content = factory.Faker("paragraph", nb_sentences=5)

    class Params:
        document_id = factory.Sequence(lambda n: n + 1)
        project_id = 1
        chunk_id = factory.Sequence(lambda n: n)
        page_number = factory.Sequence(lambda n: (n // 10) + 1)
        has_context = False
        raw_section = ""
        image_path = ""
        original_content = ""

    @factory.lazy_attribute
    def metadata(self):
        return {
            "document_id": self.document_id,
            "project_id": self.project_id,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "has_context": self.has_context,
            "raw_section": self.raw_section,
            "image_path": self.image_path,
            "original_content": self.original_content,
            "full_document_text_available": True,
        }


class ContextualizedDocumentFactory(DocumentFactory):
    """Factory for creating contextualized Document objects."""

    class Params:
        has_context = True
        context_text = factory.Faker("sentence", nb_words=10)

    @factory.lazy_attribute
    def page_content(self):
        base_content = factory.Faker("paragraph", nb_sentences=5).generate({})
        return f"<context>\n{self.context_text}\n</context>\n\n{base_content}"

    @factory.lazy_attribute
    def metadata(self):
        meta = super().metadata
        meta["has_context"] = True
        meta["context_length"] = len(self.context_text.split())
        return meta


class ChunkFactory(factory.Factory):
    """Factory for creating Chunk objects."""

    class Meta:
        model = Chunk

    splits = factory.List([factory.Faker("paragraph", nb_sentences=3)])
    is_triggered = False
    token_count = factory.LazyAttribute(lambda obj: sum(len(s.split()) * 1.3 for s in obj.splits))

    class Params:
        idx = factory.Sequence(lambda n: n)
        page_number = 1
        is_image = False

    @factory.lazy_attribute
    def metadata(self):
        meta = {
            "idx": self.idx,
            "page_number": self.page_number,
        }
        if self.is_image:
            meta["image_path"] = f"/tmp/test_image_{self.idx}.png"
        return meta

    @factory.lazy_attribute
    def content(self):
        return "\n\n".join(self.splits)


class MockDjangoDocumentFactory(factory.Factory):
    """Factory for creating mock Django document models."""

    class Meta:
        model = dict

    id = factory.Sequence(lambda n: n + 1)
    name = factory.Faker("sentence", nb_words=3)
    markdown = ""
    tokens = 0
    indexed_chunks = 0
    indexing_status = "pending"

    class Params:
        project_id = 1
        file_extension = ".pdf"
        has_file = True

    @factory.lazy_attribute
    def file(self):
        if not self.has_file:
            return None

        from unittest.mock import Mock

        mock_file = Mock()
        mock_file.name = f"{self.name.lower().replace(' ', '_')}{self.file_extension}"
        mock_file.path = f"/tmp/{mock_file.name}"
        mock_file.storage.exists.return_value = True
        return mock_file

    @factory.lazy_attribute
    def project(self):
        from unittest.mock import Mock

        mock_project = Mock()
        mock_project.id = self.project_id
        return mock_project

    @factory.post_generation
    def setup_save_method(obj, create, extracted, **kwargs):
        from unittest.mock import Mock

        obj["save"] = Mock()

        mock_obj = Mock()
        for key, value in obj.items():
            setattr(mock_obj, key, value)
        return mock_obj


class SearchResultFactory(factory.Factory):
    """Factory for creating search results."""

    class Meta:
        model = tuple

    class Params:
        score = factory.Faker("pyfloat", left_digits=0, right_digits=4, positive=True, max_value=1)

    @factory.lazy_attribute
    def document(self):
        return DocumentFactory()

    def __new__(cls, *args, **kwargs):
        doc = kwargs.get("document", DocumentFactory())
        score = kwargs.get("score", 0.95)
        return (doc, score)
