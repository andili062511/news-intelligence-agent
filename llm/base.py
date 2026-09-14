"""Model-independent language-model interface."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any


class BaseLLM(ABC):
    """Minimal interface consumed by the agent graph."""

    @abstractmethod
    def generate(self, messages: Sequence[dict[str, Any]]) -> str:
        """Generate one assistant response from chat messages."""
