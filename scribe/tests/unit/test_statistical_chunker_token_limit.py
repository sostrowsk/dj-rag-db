"""Regression test for OpenAI text-embedding-3-large 8192-token limit.

Bug staging: ProtectedClientDocument#105 indexing fails with
    Error code: 400 - Invalid 'input[38]': maximum input length is 8192 tokens.

Root cause: scribe/chunking/statisticalchunker.py._encode_documents()
passed splits straight to the encoder. SCRIBE_MAX_CHUNK_TOKENS=2999 is
respected DURING accumulation, but the upstream _max_999_splits merging
(:150-177) can produce single splits >8192 tokens after re-joining.
The fix enforces a hard cap BEFORE _encode_documents in _chunk().

Codex P1: enforcement must happen at the start of _chunk(), not inside
_encode_documents() — _split_documents() maps split_indices back onto
the same `splits` list (docs=splits), so internal expansion would
de-sync indices and corrupt chunking.
"""

from unittest.mock import MagicMock

import numpy as np
from ai_router.encoders import BaseEncoder

from scribe.chunking.statisticalchunker import StatisticalChunker_GaussianSmoothing


class _FakeEncoder(BaseEncoder):
    name: str = "fake"
    score_threshold: float = 0.5

    def __call__(self, docs):
        # Return one trivial vector per doc.
        return [np.array([1.0, 0.0]) for _ in docs]


def _make_chunker():
    enc = _FakeEncoder()
    splitter = MagicMock()
    splitter.return_value = []
    return StatisticalChunker_GaussianSmoothing(
        encoder=enc,
        splitter=splitter,
        max_split_tokens=2999,
        min_split_tokens=100,
    )


class TestEnforceEmbeddingTokenLimit:
    def test_oversized_split_is_broken_into_under_8000_token_pieces(self):
        chunker = _make_chunker()

        # 20000 short ASCII words → well over the 8192-token limit.
        oversized = " ".join([f"word{i}" for i in range(20000)])

        result = chunker._enforce_embedding_token_limit([oversized], hard_limit=8000)

        from scribe.utils import tiktoken_length

        assert all(tiktoken_length(piece) <= 8000 for piece in result)
        # Total content preserved (no silent drop)
        assert sum(len(piece.split()) for piece in result) == 20000

    def test_normal_split_is_passed_through_unchanged(self):
        chunker = _make_chunker()
        normal = " ".join([f"w{i}" for i in range(2000)])  # ~2k tokens, below limit

        result = chunker._enforce_embedding_token_limit([normal], hard_limit=8000)

        assert result == [normal]

    def test_empty_input_is_passed_through(self):
        chunker = _make_chunker()
        assert chunker._enforce_embedding_token_limit([], hard_limit=8000) == []

    def test_no_whitespace_segment_is_split_via_tokenizer(self):
        """Codex P2: doc.split(' ') alone leaves a single long no-space
        segment intact. A run of joined identifiers / one giant word /
        base64 blob can stay >8000 tokens after whitespace splitting and
        still hit OpenAI 8192. Fall back to tiktoken encode/decode
        boundary for any segment that's still oversized."""
        chunker = _make_chunker()

        from scribe.utils import tiktoken_length

        # Single "word" with no spaces, but many tokens. Repeat the same
        # subword to make sure it's tokenized into many tokens (not one
        # giant rare token).
        no_space_segment = "Antidisestablishmentarianism" * 4000

        # Sanity: the input is over the limit.
        assert tiktoken_length(no_space_segment) > 8000

        result = chunker._enforce_embedding_token_limit([no_space_segment], hard_limit=8000)

        assert all(tiktoken_length(piece) <= 8000 for piece in result)
        # Re-joining the pieces must reproduce the input exactly (no
        # silent loss of characters at boundaries).
        assert "".join(result) == no_space_segment

    def test_chunk_calls_enforce_before_encode_and_aligns_indices(self, monkeypatch):
        """Codex P1: _chunk() must enforce the limit BEFORE _encode_documents
        and pass the SAME (potentially expanded) list to _split_documents,
        so split_indices line up with docs."""
        chunker = _make_chunker()

        # Stub upstream stages so __call__ doesn't run; we test _chunk directly.
        captured = {}

        def fake_encode(docs):
            captured["encoded_len"] = len(docs)
            return np.zeros((len(docs), 2))

        def fake_split_documents(docs, split_indices, similarities):
            captured["split_documents_len"] = len(docs)
            return []

        monkeypatch.setattr(chunker, "_encode_documents", fake_encode)
        monkeypatch.setattr(chunker, "_split_documents", fake_split_documents)
        monkeypatch.setattr(
            chunker,
            "_calculate_similarity_scores",
            lambda x: [0.0] * (len(x) - 1) if len(x) > 1 else [],
        )
        # Disable the dynamic-threshold path so we don't have to stub
        # _find_optimal_threshold's full call chain.
        chunker.dynamic_threshold = False
        monkeypatch.setattr(chunker, "_find_split_indices", lambda **_: [])

        oversized = " ".join([f"word{i}" for i in range(20000)])
        normal = " ".join([f"w{i}" for i in range(500)])

        chunker._chunk([oversized, normal])

        # Both stages must see the SAME post-enforcement length so indices align.
        assert captured["encoded_len"] == captured["split_documents_len"]
        # Oversized split must have been broken up → more than 2 docs
        assert captured["encoded_len"] > 2
