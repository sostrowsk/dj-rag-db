from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    splits: List[str] = Field(default_factory=list)
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        if self.content:
            return self.content
        return " ".join(self.splits) if self.splits else ""

    @property
    def text(self) -> str:
        return str(self)

    class Config:
        arbitrary_types_allowed = True
