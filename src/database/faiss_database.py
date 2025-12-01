import numpy as np
import faiss


class FaissDatabase:
    """
    A vector database using FAISS for similarity search.
    
    Uses IndexFlatL2 for exact L2 distance search.
    Stores chunk metadata alongside embeddings.
    """

    def __init__(self, dimension: int):
        """
        Initialize the FAISS database.
        
        Args:
            dimension: The dimensionality of the embedding vectors.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: list[dict[str, str]] = []

    def insert_embeddings(self, embeddings_matrix: np.ndarray[np.float32], chunks: list[dict[str, str]]) -> None:
        """
        Insert embeddings and their associated chunks into the database.

        Args:
            embeddings_matrix: Matrix of float32 embedding vectors.
            chunks: List of chunk metadata dictionaries corresponding to each embedding.
        """
        self.index.add(embeddings_matrix)
        self.chunks.extend(chunks)

    def search(self, query: np.ndarray[np.float32], k: int = 5) -> list[dict[str, str]]:
        """
        Search for the k most similar chunks to the query embedding.

        Args:
            query: The float32 query vector to search for.
            k: Number of nearest neighbors to return.

        Returns:
            List of chunk metadata dictionaries for the k most similar chunks.
        """
        if self.index.ntotal == 0:
            return []

        # Validate query is 2D float32 array with shape (1, dimension)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("query must be 2D with shape (1, dimension)")

        # Limit k to available vectors
        k = min(k, self.index.ntotal)

        # Search returns distances and indices
        distances, indices = self.index.search(query, k)

        # Retrieve corresponding chunks
        results = []
        for idx in indices[0]:
            if idx >= 0:  # FAISS returns -1 for unfilled slots
                results.append(self.chunks[idx])

        return results

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors in the index."""
        return self.index.ntotal

    def clear(self) -> None:
        """Clear all vectors and chunks from the database."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
