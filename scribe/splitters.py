from abc import ABC, abstractmethod
from typing import List


class BaseSplitter(ABC):
    @abstractmethod
    def __call__(self, doc: str) -> List[str]:
        pass
