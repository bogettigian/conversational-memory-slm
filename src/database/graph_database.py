import uuid

import networkx
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from openie import StanfordOpenIE
from sentence_transformers import SentenceTransformer

from src.database.faiss_database import FaissDatabase


class GraphDatabase:

    def __init__(self, embeddings_model_name: str, vector_db: FaissDatabase):
        nltk.download("punkt_tab")

        self.graph = networkx.Graph()
        self.embedding_model = SentenceTransformer(embeddings_model_name, trust_remote_code=True)
        self.vector_db = vector_db

    def _insert_attr_node(self, attr: str, parent_id: str, attr_type: str):
        attr_embedding = self.embedding_model.encode(attr, convert_to_numpy=True).astype(np.float32).reshape(1, -1)
        result_attr, m_attr = self.vector_db.search(attr_embedding, k=1, threshold=0.95)
        if len(result_attr) == 0:
            attr_id = str(uuid.uuid4())
            self.graph.add_node(attr_id)
            self.graph.nodes[attr_id]["data"] = attr
            self.graph.nodes[attr_id]["type"] = attr_type
            self.graph.nodes[attr_id]["metadata"] = {}
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

                sentences = sent_tokenize(chunk, language="english")
                for sentence in [s.strip() for s in sentences if s.strip()]:
                    sentence_id = str(uuid.uuid4())
                    self.graph.add_node(sentence_id)
                    self.graph.nodes[sentence_id]["data"] = sentence
                    self.graph.nodes[sentence_id]["type"] = "sentence"
                    self.graph.nodes[sentence_id]["metadata"] = {}

                    self.graph.add_edge(m["id"], sentence_id)

                    for extraction in client.annotate(sentence):
                        self._insert_attr_node(extraction["subject"], sentence_id, "subject")
                        self._insert_attr_node(extraction["relation"], sentence_id, "relation")
                        self._insert_attr_node(extraction["object"], sentence_id, "object")

    def search_nodes(self, ids: list[str]) -> tuple[list[str], list[dict[str, str]]]:
        result_ids = []
        result_chunks = []
        result_metadata = []
        for id in ids:
            for sentence_adj in list(self.graph.adj[id]):
                for attr_adj in list(self.graph.adj[sentence_adj]):
                    if self.graph.nodes[attr_adj]["type"] != "chunk":
                        for adj in list(self.graph.adj[attr_adj]):
                                for chunk_adj in list(self.graph.adj[adj]):
                                    if self.graph.nodes[chunk_adj]["type"] == "chunk" and chunk_adj not in result_ids:
                                        result_ids.append(chunk_adj)
                                        result_chunks.append(self.graph.nodes[chunk_adj]["data"])
                                        result_metadata.append(self.graph.nodes[chunk_adj]["metadata"])
        return result_chunks, result_metadata

    def clear(self):
        self.graph.clear()
        self.vector_db.clear()