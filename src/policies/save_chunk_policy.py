import math
from abc import ABC, abstractmethod
from collections import deque

from transformers import AutoTokenizer

from src.datasets.dataset import Session


SEARCH_PREFIX = "search_document: "


class SaveChunkPolicy(ABC):
    name: str
    @abstractmethod
    def apply(self, session_history: list[Session]) -> list[str]:
        """Partition the session history into chunks, so we can later embed them and save them in the database.
        """
        pass


class SlidingWindowSaveChunkPolicy(SaveChunkPolicy):
    """Join messages into chunks that fit in the max chunk length and max chunk overlap.
    
    If it contains a message that exceeds the max chunk length, it will be split into smaller messages that fit in the chunk size."""
    name = "sliding_window"

    def __init__(self, max_chunk_length: int, max_chunk_overlap: int):
        self.max_chunk_length = max_chunk_length
        self.max_chunk_overlap = max_chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    @staticmethod
    def _format_message(message: dict) -> str:
        role = message["role"].capitalize()
        content = message["content"]
        return f"{role}: {content}"

    @staticmethod
    def _format_chunk(messages: list[dict]) -> str:
        dialogue_lines = [SlidingWindowSaveChunkPolicy._format_message(msg) for msg in messages]
        return SEARCH_PREFIX + "\n".join(dialogue_lines)

    def _split_messages_by_tokens(self, session: Session) -> list[dict]:
        """Split messages that exceed the max token budget for a single chunk."""
        metadata_tokens = len(self.tokenizer.encode(SEARCH_PREFIX))
        available_tokens = self.max_chunk_length - metadata_tokens

        split_messages = []
        for message in session.messages:
            formatted = self._format_message(message)
            tokens = self.tokenizer.encode(formatted)

            if len(tokens) <= available_tokens:
                split_messages.append(message)
            else:
                # Split long message into smaller messages that fit in the chunk size
                role = message["role"].capitalize()
                content = message["content"]
                content_tokens = self.tokenizer.encode(content)

                role_overhead = len(self.tokenizer.encode(f"{role}: "))
                ellipsis_overhead = len(self.tokenizer.encode("..."))
                max_content_tokens = available_tokens - role_overhead - ellipsis_overhead

                num_chunks = math.ceil(len(content_tokens) / max_content_tokens)
                for j, start in enumerate(range(0, len(content_tokens), max_content_tokens)):
                    chunk_tokens = content_tokens[start: start + max_content_tokens]
                    chunk_content = self.tokenizer.decode(chunk_tokens)

                    # Add ellipsis for split messages
                    is_first = (j == 0)
                    is_last = (j == num_chunks - 1)
                    if not is_first:
                        chunk_content = "..." + chunk_content
                    if not is_last:
                        chunk_content = chunk_content + "..."

                    split_messages.append({"role": role, "content": chunk_content})

        return split_messages

    def _create_token_bounded_chunks(self, messages: list[dict]) -> list[str]:
        """Create chunks that respect max_chunk_length and max_chunk_overlap in tokens."""
        metadata_tokens = len(self.tokenizer.encode(SEARCH_PREFIX))
        available_tokens = self.max_chunk_length - metadata_tokens

        chunks = []
        current_messages = []
        current_token_count = 0

        overlap_window = deque()
        overlap_token_count = 0

        for message in messages:
            formatted = self._format_message(message)
            msg_tokens = len(self.tokenizer.encode(formatted + "\n"))

            if current_token_count + msg_tokens <= available_tokens:
                current_messages.append(message)
                current_token_count += msg_tokens
            else:
                if current_messages:
                    chunks.append(self._format_chunk(current_messages))

                current_messages = [msg for msg, _ in overlap_window] + [message]
                current_token_count = overlap_token_count + msg_tokens

            overlap_window.append((message, msg_tokens))
            overlap_token_count += msg_tokens

            while overlap_token_count > self.max_chunk_overlap and overlap_window:
                _, removed_tokens = overlap_window.popleft()
                overlap_token_count -= removed_tokens

        if current_messages:
            chunks.append(self._format_chunk(current_messages))

        return chunks

    def apply(self, session_history: list[Session]) -> list[str]:
        chunks = []

        for session in session_history:
            split_messages = self._split_messages_by_tokens(session)
            session_chunks = self._create_token_bounded_chunks(split_messages)
            chunks.extend(session_chunks)

        return chunks
