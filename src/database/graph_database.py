import uuid

import networkx
import numpy as np
from openie import StanfordOpenIE
from sentence_transformers import SentenceTransformer

from src.database.faiss_database import FaissDatabase


class GraphDatabase:

    def __init__(self, embeddings_model_name: str, vector_db: FaissDatabase):
        self.graph = networkx.Graph()
        self.embedding_model = SentenceTransformer(embeddings_model_name, trust_remote_code=True)
        self.vector_db = vector_db

    def _insert_attr_node(self, attr: str, parent_id: str, attr_type: str):
        attr_embedding = self.embedding_model.encode(attr, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        result_attr, m_attr = self.vector_db.search(attr_embedding, k=1, threshold=0.95)
        if len(result_attr) == 0:
            attr_id = str(uuid.uuid4())
            metadata = {"role": "user", "id": attr_id}
            self.graph.add_node(attr_id)
            self.graph.nodes[attr_id]["data"] = attr
            self.graph.nodes[attr_id]["type"] = attr_type
            self.graph.nodes[attr_id]["metadata"] = metadata

            self.vector_db.insert_embeddings(np.array([attr_embedding]), [attr], [metadata])
        else:
            attr_id = m_attr[0]["id"]
        self.graph.add_edge(parent_id, attr_id)

    def insert_nodes(
            self,
            chunks: list[str],
            metadata: list[dict[str, str]]
    ) -> None:
        with StanfordOpenIE() as client:
            for chunk, m in zip(chunks, metadata):
                self.graph.add_node(m["id"])
                self.graph.nodes[m["id"]]["data"] = chunk
                self.graph.nodes[m["id"]]["type"] = "chunk"
                self.graph.nodes[m["id"]]["metadata"] = m

                for extraction in client.annotate(chunk):
                    self._insert_attr_node(extraction["subject"], m["id"], "subject")
                    self._insert_attr_node(extraction["relation"], m["id"], "relation")
                    self._insert_attr_node(extraction["object"], m["id"], "object")

    def search_nodes(self, ids: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        result_ids = set()
        result_chunks = []
        result_metadata = []
        # For each chunk, lookup its relevant relations and add all directly connected chunks to these relations
        for id in ids:
            for attr_adj in list(self.graph.adj[id]):
                for adj in list(self.graph.adj[attr_adj]):
                    if self.graph.nodes[adj]["type"] == "chunk" and adj not in result_ids:
                        result_ids.add(adj)
                        result_chunks.append(self.graph.nodes[adj]["data"])
                        result_metadata.append(self.graph.nodes[adj]["metadata"])
        return result_chunks, result_metadata

    def clear(self):
        self.graph.clear()
        self.vector_db.clear()