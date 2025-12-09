import numpy as np
import faiss


class FaissDatabase:
    """
    A vector database using FAISS for similarity search.
    
    Uses IndexFlatIP with L2-normalized vectors for cosine similarity search.
    Stores chunk metadata alongside embeddings.
    """

    def __init__(self, dimension: int):
        """
        Initialize the FAISS database.
        
        Args:
            dimension: The dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self.index_user = faiss.IndexFlatIP(dimension)
        self.index_assistant = faiss.IndexFlatIP(dimension)
        self.index_all = faiss.IndexFlatIP(dimension)
        self.chunks_user: list[str] = []
        self.metadata_user: list[dict] = []
        self.chunks_assistant: list[str] = []
        self.metadata_assistant: list[dict] = []
        self.chunks_all: list[str] = []
        self.metadata_all: list[dict] = []

    def _get_store(self, role: str):
        normalized_role = role.lower()
        if normalized_role == "user":
            return self.index_user, self.chunks_user, self.metadata_user
        if normalized_role == "assistant":
            return self.index_assistant, self.chunks_assistant, self.metadata_assistant
        if normalized_role == "all":
            return self.index_all, self.chunks_all, self.metadata_all
        raise ValueError("role must be 'user' or 'assistant'")

    def insert_embeddings(
            self,
            embeddings_matrix: np.ndarray[np.float32],
            chunks: list[str],
            metadata: list[dict],
    ) -> None:
        """
        Insert embeddings and their associated chunks into the database.

        Args:
            embeddings_matrix: Matrix of float32 embedding vectors.
            chunks: List of chunk strings corresponding to each embedding.
            metadata: List of metadata dictionaries corresponding to each embedding.
        """
        if embeddings_matrix.shape[0] != len(chunks) or len(chunks) != len(metadata):
            raise ValueError("embeddings_matrix, chunks, and metadata must have the same length")

        roles = [meta["role"] for meta in metadata]

        grouped_embeddings = {"user": [], "assistant": [], "all": []}
        grouped_chunks = {"user": [], "assistant": [], "all": []}
        grouped_metadata = {"user": [], "assistant": [], "all": []}

        for embedding, chunk, meta, role in zip(embeddings_matrix, chunks, metadata, roles):
            normalized_chunk = chunk[len("search_document: "):] if chunk.startswith("search_document: ") else chunk
            if role not in grouped_embeddings:
                raise ValueError("role must be 'user' or 'assistant'")

            grouped_embeddings[role].append(embedding.astype(np.float32))
            grouped_chunks[role].append(normalized_chunk)
            grouped_metadata[role].append(meta)

            grouped_embeddings["all"].append(embedding.astype(np.float32))
            grouped_chunks["all"].append(normalized_chunk)
            grouped_metadata["all"].append(meta)

        for role in ("user", "assistant", "all"):
            if not grouped_embeddings[role]:
                continue

            embeddings_batch = np.vstack(grouped_embeddings[role]).astype(np.float32)
            faiss.normalize_L2(embeddings_batch)
            index, chunk_store, metadata_store = self._get_store(role)

            index.add(embeddings_batch)
            chunk_store.extend(grouped_chunks[role])
            metadata_store.extend(grouped_metadata[role])

    def search(self, query: np.ndarray[np.float32], k: int, threshold: float, role: str = "all") -> tuple[list[str], list[dict]]:
        """
        Search for the k most similar chunks to the query embedding.

        Args:
            role: Which index to search. Accepted values: 'user' or 'assistant'.
            query: The float32 query vector to search for.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (list of chunk strings, list of metadata) for the k most similar chunks that have cosine similarity greater than or equal to threshold.
        """
        index, chunks, metadata_store = self._get_store(role)
        if index.ntotal == 0:
            return [], []

        # Validate query is 2D float32 array with shape (1, dimension)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query must be 2D with shape (1, dimension)")

        faiss.normalize_L2(query)

        # Limit k to available vectors
        k = min(k, index.ntotal)

        # Search returns cosine similarities and indices
        similarities, indices = index.search(query, k)

        # Retrieve corresponding chunks if cosine similarity is greater than or equal to threshold
        results = []
        results_metadata = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and similarities[0][i] >= threshold:  # FAISS returns index=-1 for unfilled slots
                results.append(chunks[idx])
                results_metadata.append(metadata_store[idx])

        return results, results_metadata

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors in the index."""
        return self.index_user.ntotal + self.index_assistant.ntotal

    def clear(self) -> None:
        """Clear all vectors and chunks from the database."""
        self.index_user = faiss.IndexFlatIP(self.dimension)
        self.index_assistant = faiss.IndexFlatIP(self.dimension)
        self.chunks_user = []
        self.metadata_user = []
        self.chunks_assistant = []
        self.metadata_assistant = []
