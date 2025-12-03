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
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[str] = []
        self.metadata: list[dict] = []

    def insert_embeddings(self, embeddings_matrix: np.ndarray[np.float32], chunks: list[str], metadata: list[dict]) -> None:
        """
        Insert embeddings and their associated chunks into the database.

        Args:
            embeddings_matrix: Matrix of float32 embedding vectors.
            chunks: List of chunk strings corresponding to each embedding.
        """
        faiss.normalize_L2(embeddings_matrix)
        self.index.add(embeddings_matrix)
        normalized_chunks = []
        for chunk in chunks:
            if chunk.startswith("search_document: "):
                normalized_chunks.append(chunk[len("search_document: "):])
            else:
                normalized_chunks.append(chunk)
        self.chunks.extend(normalized_chunks)
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray[np.float32], k: int = 5, threshold: float = 0.6) -> tuple[list[str], list[dict]]:
        """
        Search for the k most similar chunks to the query embedding.

        Args:
            query: The float32 query vector to search for.
            k: Number of nearest neighbors to return.

        Returns:
            Tuple of (list of chunk strings, list of metadata) for the k most similar chunks that have cosine similarity greater than or equal to threshold.
        """
        if self.index.ntotal == 0:
            return []

        # Validate query is 2D float32 array with shape (1, dimension)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query must be 2D with shape (1, dimension)")

        faiss.normalize_L2(query)

        # Limit k to available vectors
        k = min(k, self.index.ntotal)

        # Search returns cosine similarities and indices
        similarities, indices = self.index.search(query, k)

        # Retrieve corresponding chunks if cosine similarity is greater than or equal to threshold
        results = []
        metadata = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and similarities[0][i] >= threshold:  # FAISS returns index=-1 for unfilled slots
                results.append(self.chunks[idx])
                metadata.append(self.metadata[idx])

        return results, metadata

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors in the index."""
        return self.index.ntotal

    def clear(self) -> None:
        """Clear all vectors and chunks from the database."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.metadata = []
