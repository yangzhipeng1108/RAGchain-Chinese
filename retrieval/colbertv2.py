from typing import List, Union
from uuid import UUID

from langchain.schema import Document
from langchain.schema.vectorstore import VectorStore

from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.vectorstore.base import SlimVectorStore
import requests
from typing import List, Dict
from ragatouille import RAGPretrainedModel

class ColbertV2Retrieval(BaseRetrieval):
    """
    VectorDBRetrieval is a retrieval class that uses VectorDB as a backend.
    First, embed the passage content using an embedding model.
    Then, store the embedded vector in VectorDB.
    When retrieving, embed the query and search the most similar vectors in VectorDB.
    Lastly, return the passages that have the most similar vectors.
    """

    def __init__(self, model_path="colbert-ir/colbertv2.0"):
        """
        :param vectordb: VectorStore instance. You can all langchain VectorStore classes, also you can use SlimVectorStore for better storage efficiency.
        """
        super().__init__()
        self.RAG = RAGPretrainedModel.from_pretrained(model_path)

    def ingest(self, passages: List[Passage]):
        self.RAG.index(
            collection=[passages],
            index_name="Miyazaki",
            max_document_length=180,
            split_documents=True,
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Passage]:
        ids = self.retrieve_id(query, top_k)
        passage_list = self.fetch_data(ids)
        return passage_list

    def retrieve_id(self, query: str, top_k: int = 5) -> List[Union[str, UUID]]:
        docs = self.rag.search(query=query, k=top_k)
        return [self.__str_to_uuid(doc.metadata.get('passage_id')) for doc in docs]

    def retrieve_id_with_scores(self, query: str, top_k: int = 5) -> tuple[
        List[Union[str, UUID]], List[float]]:
        results = self.vectordb.similarity_search_with_score(query=query, k=top_k)
        results = results[::-1]
        docs = [result[0] for result in results]
        scores = [result[1] for result in results]
        return [self.__str_to_uuid(doc.metadata.get('passage_id')) for doc in docs], scores

    def delete(self, ids: List[Union[str, UUID]]):
        self.vectordb.delete([str(_id) for _id in ids])

    @staticmethod
    def __str_to_uuid(input_str: str) -> Union[str, UUID]:
        try:
            return UUID(input_str)
        except:
            return input_str
