from abc import ABC, abstractmethod


class OutputTruncatedError(RuntimeError):
    """The model stopped because it hit its output-token limit."""


class ConsistencyEditorProvider(ABC):
    """A chat-completion backend used for the chapter-level consistency/editing pass.

    Implementations wrap a specific API surface (OpenAI-compatible, Anthropic, ...).
    The consistency-editing prompt logic stays provider-agnostic and only depends
    on this interface, so swapping models/backends never touches pipeline code.
    """

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Run one turn and return the model's plain-text reply."""
        raise NotImplementedError