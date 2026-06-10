import logging
from typing import List

import numpy as np
from ai_router.encoders import BaseEncoder
from scipy.ndimage import gaussian_filter1d

from scribe.chunkers import BaseChunker
from scribe.schema import Chunk
from scribe.splitters import BaseSplitter
from scribe.utils import tiktoken_length, time_it

from ..tools.md_regex_splitter import MDRegexSplitter

logger = logging.getLogger(__name__)


class StatisticalChunker_GaussianSmoothing(BaseChunker):
    encoder: BaseEncoder

    def __init__(
        self,
        encoder: BaseEncoder,
        splitter: BaseSplitter = MDRegexSplitter(),
        name="statistical_chunker",
        threshold_adjustment=0.01,
        dynamic_threshold: bool = True,
        window_size=20,
        min_split_tokens=100,
        max_split_tokens=300,
        split_tokens_tolerance=10,
        gaussian_sigma=1.0,
    ):
        super().__init__(name=name, encoder=encoder, splitter=splitter)
        self.encoder = encoder
        self.threshold_adjustment = threshold_adjustment
        self.dynamic_threshold = dynamic_threshold
        self.window_size = window_size
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.DEFAULT_THRESHOLD = 0.5
        self.gaussian_sigma = gaussian_sigma

    @time_it
    def __call__(self, docs: List[str]) -> List[Chunk]:
        if not docs:
            raise ValueError("At least one document is required for splitting.")

        all_chunks = []
        for doc in docs:
            splits = self._split(doc)
            joined_splits = self._join_splits(splits)
            cleaned_splits = self._clean_splits(joined_splits)
            final_splits = self._max_999_splits(cleaned_splits)
            doc_chunks = self._chunk(final_splits)
            all_chunks.append(doc_chunks)
        return all_chunks

    # OpenAI text-embedding-3-large hard limit is 8192 tokens per input.
    # A safety margin (8000) leaves headroom for tokenizer drift and for
    # the BPE encoder treating a few characters differently than tiktoken.
    EMBEDDING_HARD_TOKEN_LIMIT = 8000

    def _enforce_embedding_token_limit(
        self, docs: List[str], hard_limit: int = EMBEDDING_HARD_TOKEN_LIMIT
    ) -> List[str]:
        """Last-resort guard before _encode_documents.

        SCRIBE_MAX_CHUNK_TOKENS=2999 is respected during accumulation in
        _split_documents, but _max_999_splits / _join_splits / _clean_splits
        can re-merge splits past that boundary and produce a single split
        > the embedding API limit (8192 tokens for text-embedding-3-large).
        OpenAI returns 400 'Invalid input[i]: maximum input length is
        8192 tokens.' which surfaces as a fatal indexing failure.

        Two-tier split:
        1. Whitespace boundaries (preserves word semantics where possible).
        2. Tokenizer-boundary fallback for any remaining oversized piece —
           a long no-space segment (joined identifiers, base64 blob, one
           giant word) would otherwise survive the whitespace pass and
           still trip the embedding limit (Codex P2).
        Token count via the existing tiktoken_length helper (BaseEncoder
        has no tokenizer API of its own).
        """
        if not docs:
            return []
        result: List[str] = []
        for doc in docs:
            if tiktoken_length(doc) <= hard_limit:
                result.append(doc)
                continue
            for piece in self._split_on_whitespace(doc, hard_limit):
                if tiktoken_length(piece) <= hard_limit:
                    result.append(piece)
                else:
                    # Whitespace pass left an oversized piece (no spaces
                    # to split on, or a single very long token). Fall
                    # back to tokenizer-boundary slicing.
                    result.extend(self._split_on_token_boundary(piece, hard_limit))
        return result

    @staticmethod
    def _split_on_whitespace(doc: str, hard_limit: int) -> List[str]:
        """Greedy whitespace-bounded chunking. May still produce
        oversized pieces when the source has no spaces — the caller
        applies a tokenizer-boundary fallback for those."""
        words = doc.split(" ")
        result: List[str] = []
        chunk_words: List[str] = []
        for w in words:
            candidate = (" ".join(chunk_words + [w])) if chunk_words else w
            if chunk_words and tiktoken_length(candidate) > hard_limit:
                result.append(" ".join(chunk_words))
                chunk_words = [w]
            else:
                chunk_words.append(w)
        if chunk_words:
            result.append(" ".join(chunk_words))
        return result

    @staticmethod
    def _split_on_token_boundary(text: str, hard_limit: int) -> List[str]:
        """Split via tiktoken encode/decode so each piece is <= hard_limit
        tokens. Used as a last-resort for segments without whitespace.
        Concatenating the result reproduces the input exactly (decode is
        the inverse of encode for cl100k_base on valid UTF-8)."""
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model("gpt-5")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        token_ids = encoding.encode(text)
        result: List[str] = []
        for start in range(0, len(token_ids), hard_limit):
            slice_ids = token_ids[start : start + hard_limit]
            result.append(encoding.decode(slice_ids))
        return result

    @time_it
    def _chunk(self, splits: List[str]) -> List[Chunk]:
        # Codex P1: enforce BEFORE encode and BEFORE _split_documents so
        # the post-enforcement list is what gets indexed and similarity-
        # mapped — otherwise split_indices would point into the wrong
        # docs and corrupt chunk boundaries.
        splits = self._enforce_embedding_token_limit(splits)
        encoded_splits = self._encode_documents(splits)
        similarities = self._calculate_similarity_scores(encoded_splits)

        if self.dynamic_threshold:
            calculated_threshold = self._find_optimal_threshold(splits, similarities)
        else:
            calculated_threshold = (
                self.encoder.score_threshold if self.encoder.score_threshold else self.DEFAULT_THRESHOLD
            )

        split_indices = self._find_split_indices(similarities=similarities, calculated_threshold=calculated_threshold)

        return self._split_documents(
            docs=splits,
            split_indices=split_indices,
            similarities=similarities,
        )

    @time_it
    def _encode_documents(self, docs: List[str]) -> np.ndarray:
        try:
            embeddings = self.encoder(docs)
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error encoding documents: {str(e)}")
            raise

    def _calculate_similarity_scores(self, encoded_docs: np.ndarray) -> List[float]:
        raw_similarities = []
        window_size = self.window_size

        for idx in range(1, len(encoded_docs)):
            window_start = max(0, idx - window_size)

            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            curr_sim_score = np.dot(cumulative_context, encoded_docs[idx]) / (
                np.linalg.norm(cumulative_context) * np.linalg.norm(encoded_docs[idx]) + 1e-10
            )
            raw_similarities.append(curr_sim_score)
        smoothed_similarities = gaussian_filter1d(raw_similarities, sigma=self.gaussian_sigma)
        return smoothed_similarities.tolist()

    def _join_splits(self, splits):
        result = []
        for i, split in enumerate(splits):
            if i == 0:
                result.append(split)
                continue
            prev_split = result[-1]
            if len(split) > 2 and len(prev_split) > 2 and prev_split[-2:-1] == "**" and split[0:1] == "**":
                result[-1] = prev_split[:-2] + " " + split[2:]
            elif len(split) > 2 and len(prev_split) > 2 and prev_split[-2:-1] == "~~" and split[0:1] == "~~":
                result[-1] = prev_split[:-2] + " " + split[2:]
            elif len(split) > 1 and len(prev_split) > 1 and prev_split[-1] == "*" and split[0] == "*":
                result[-1] = prev_split[:-1] + " " + split[1:]
            elif len(split) > 1 and len(prev_split) > 1 and split[0].islower():
                result[-1] += " " + split
            elif (
                len(split) > 1
                and len(prev_split) > 1
                and prev_split[-1].isalnum()
                and not split[0] in ["_", "#", "`", ">", "!"]
            ):
                result[-1] += " " + split
            elif (
                len(split) > 1
                and len(prev_split) > 1
                and prev_split[-1] in [",", ";", ":", "-"]
                and not split[0].isalnum()
            ):
                result[-1] += " " + split
            else:
                result.append(split)
        return result

    def _clean_splits(self, splits):
        results = []
        current_split = ""
        for split in splits:
            if split.isdigit() or len(set(split)) > 1 and len(split) > 1:
                current_split += " " + split
            else:
                logger.debug("Ignored split: %s", split)
            if tiktoken_length(current_split) > 10:
                results.append(current_split)
                current_split = ""
        return results

    def _max_999_splits(self, splits):
        if len(splits) <= 999:
            return splits
        min_tokens = 20
        while len(splits) > 999:
            merged_splits = []
            current_split = ""
            for part in splits:
                part_tokens = tiktoken_length(part)
                if part_tokens > min_tokens:
                    if current_split:
                        merged_splits.append(current_split)
                        current_split = ""
                    merged_splits.append(part)
                else:
                    new_split = current_split + (" " if current_split else "") + part
                    new_tokens = tiktoken_length(new_split)
                    if self.max_split_tokens and new_tokens > self.max_split_tokens:
                        if current_split:
                            merged_splits.append(current_split)
                        current_split = part
                    else:
                        current_split = new_split
            if current_split:
                merged_splits.append(current_split)
            splits = merged_splits
            min_tokens += 10
        return splits

    def _find_split_indices(self, similarities: List[float], calculated_threshold: float) -> List[int]:
        split_indices = []
        for idx, score in enumerate(similarities):
            if idx < 1 or idx >= len(similarities) - 1:
                continue
            if score < calculated_threshold and score < similarities[idx - 1] and score < similarities[idx + 1]:
                split_indices.append(idx + 1)
        return split_indices

    def _find_optimal_threshold(self, docs: List[str], similarity_scores: List[float]) -> float:
        token_counts = [tiktoken_length(doc) for doc in docs]
        cumulative_token_counts = np.cumsum([0] + token_counts)

        median_score = np.median(similarity_scores)
        std_dev = np.std(similarity_scores)
        low = max(0.0, float(median_score - std_dev))
        high = min(1.0, float(median_score + std_dev))

        iteration = 0
        calculated_threshold = (low + high) / 2
        while low <= high:
            split_indices = self._find_split_indices(similarity_scores, calculated_threshold)

            split_token_counts = [
                cumulative_token_counts[end] - cumulative_token_counts[start]
                for start, end in zip([0] + split_indices, split_indices + [len(token_counts)])
            ]

            median_tokens = np.median(split_token_counts)
            if (
                self.min_split_tokens - self.split_tokens_tolerance
                <= median_tokens
                <= self.max_split_tokens + self.split_tokens_tolerance
            ):
                break
            elif median_tokens < self.min_split_tokens:
                high = calculated_threshold - self.threshold_adjustment
            else:
                low = calculated_threshold + self.threshold_adjustment
            calculated_threshold = (low + high) / 2
            iteration += 1

        return calculated_threshold

    def _split_documents(self, docs: List[str], split_indices: List[int], similarities: List[float]) -> List[Chunk]:
        chunks = []
        current_split = []
        current_tokens_count = 0
        if len(docs) <= 1:
            self.min_split_tokens = 10

        for doc_idx, doc in enumerate(docs):
            doc_token_count = tiktoken_length(doc)

            if doc_idx + 1 in split_indices:
                if self.min_split_tokens <= current_tokens_count + doc_token_count < self.max_split_tokens:
                    current_split.append(doc)
                    current_tokens_count += doc_token_count
                    triggered_score = similarities[doc_idx] if doc_idx < len(similarities) else None
                    chunks.append(
                        Chunk(
                            splits=current_split.copy(),
                            is_triggered=True,
                            triggered_score=triggered_score,
                            token_count=current_tokens_count,
                        )
                    )
                    current_split, current_tokens_count = [], 0
                    continue

            if current_tokens_count + doc_token_count > self.max_split_tokens:
                if current_tokens_count >= self.min_split_tokens:
                    chunks.append(
                        Chunk(
                            splits=current_split.copy(),
                            is_triggered=False,
                            triggered_score=None,
                            token_count=current_tokens_count,
                        )
                    )
                    current_split, current_tokens_count = [], 0

            current_split.append(doc)
            current_tokens_count += doc_token_count

        if current_split:
            chunks.append(
                Chunk(
                    splits=current_split.copy(),
                    is_triggered=False,
                    triggered_score=None,
                    token_count=current_tokens_count,
                )
            )

        return chunks
