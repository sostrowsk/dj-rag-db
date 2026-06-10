from abc import ABC, abstractmethod
from typing import List

from scribe.schema import Chunk
from scribe.splitters import BaseSplitter


class BaseChunker(ABC):
    def __init__(self, name: str, encoder=None, splitter: BaseSplitter = None):
        self.name = name
        self.encoder = encoder
        self.splitter = splitter

    @abstractmethod
    def __call__(self, docs: List[str]) -> List[Chunk]:
        pass

    def _split(self, doc: str) -> List[str]:
        if self.splitter:
            return self.splitter(doc)
        return [doc]
