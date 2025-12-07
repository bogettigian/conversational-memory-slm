from abc import ABC, abstractmethod

from litellm import completion
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.datasets.dataset import Session
from src.utils.prompt import get_contextual_prompt


def _format_message(message: dict) -> str:
    role = message["role"].capitalize()
    content = message["content"]
    return f"{role}: {content}"


def _split_messages_by_tokens(
    tokenizer: PreTrainedTokenizerFast,
    session: Session,
    max_chunk_length: int,
    max_chunk_overlap: int,
) -> list[dict]:
    """Split messages that exceed the max token budget for a single chunk.

    When a turn needs to be broken into multiple chunks, consecutive chunks will
    overlap by up to ``max_chunk_overlap`` tokens to retain local continuity.
    """
    split_messages = []
    for message in session.messages:
        formatted = _format_message(message)
        tokens = tokenizer.encode(formatted)

        if len(tokens) <= max_chunk_length:
            split_messages.append({"role": message["role"], "content": formatted})
        else:
            step = max_chunk_length - max_chunk_overlap
            if step <= 0:
                raise ValueError(
                    "max_chunk_overlap must be smaller than max_chunk_length"
                )

            for start in range(0, len(tokens), step):
                end = min(start + max_chunk_length, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = tokenizer.decode(chunk_tokens)
                if start > 0:
                    chunk_text = "..." + chunk_text
                if end < len(tokens):
                    chunk_text = chunk_text + "..."
                split_messages.append({"role": message["role"], "content": chunk_text})
    return split_messages


class SaveChunkPolicy(ABC):
    name: str

    @abstractmethod
    def apply(self, session_history: list[Session]) -> tuple[list[str], list[dict]]:
        """Partition the session history into chunks, so we can later embed them and save them in the database."""
        pass


class SlidingWindowSaveChunkPolicy(SaveChunkPolicy):
    """Chunk the conversation with a sliding window respecting the token budget.

    Each chunk contains at most one full turn (user or assistant). A turn is
    only broken into multiple chunks if it individually exceeds the maximum
    chunk length."""

    name = "sliding_window"

    def __init__(
        self,
        max_chunk_length: int,
        max_chunk_overlap: int,
        tokenizer_model_name: str,
    ):
        self.max_chunk_length = max_chunk_length
        self.max_chunk_overlap = max_chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=True)

    def apply(self, session_history: list[Session]) -> tuple[list[str], list[dict]]:
        chunks = []
        metadata = []

        for session in session_history:
            split_messages = _split_messages_by_tokens(
                self.tokenizer, session, self.max_chunk_length, self.max_chunk_overlap
            )
            chunks.extend([message["content"] for message in split_messages])
            metadata.extend(
                [
                    {"date": session.date, "role": message["role"].lower()}
                    for message in split_messages
                ]
            )

        return chunks, metadata


class ContextualSlidingWindowSaveChunkPolicy(SaveChunkPolicy):
    """Chunk the conversation with a sliding window respecting the token budget.

    Each chunk contains at most one full turn (user or assistant). A turn is
    only broken into multiple chunks if it individually exceeds the maximum
    chunk length."""

    name = "contextual_sliding_window"

    def __init__(
        self,
        contextual_model_name: str,
        max_chunk_length: int,
        max_chunk_overlap: int,
        tokenizer_model_name: str,
    ):
        self.contextual_model_name = contextual_model_name
        self.max_chunk_length = max_chunk_length
        self.max_chunk_overlap = max_chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=True)

    def apply(self, session_history: list[Session]) -> tuple[list[str], list[dict]]:
        chunks = []
        metadata = []

        for session in session_history:
            split_messages = _split_messages_by_tokens(
                self.tokenizer, session, self.max_chunk_length, self.max_chunk_overlap
            )
            metadata.extend(
                [
                    {"date": session.date, "role": message["role"].lower()}
                    for message in split_messages
                ]
            )

            contextual_session_chunks = []
            for message in split_messages:
                prompt = get_contextual_prompt(session, message["content"])
                response = completion(model=self.contextual_model_name, messages=prompt)
                contextual_session_chunks.append(
                    f"{response.choices[0].message.content.strip()} {message['content']}"
                )

            chunks.extend(contextual_session_chunks)

        return chunks, metadata
