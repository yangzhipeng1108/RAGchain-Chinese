import requests
from typing import List, Dict
from ragatouille import RAGPretrainedModel

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class CustomRetriever(BaseRetriever):
    rag: RAGPretrainedModel
    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.rag.search(query=query, k=self.k)
        return [Document(page_content=doc["content"]) for doc in results]

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

def get_rag() -> RAGPretrainedModel:
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    return RAG

def rag_index(RAG:RAGPretrainedModel) -> RAGPretrainedModel:
    full_document = get_wikipedia_page("Hayao_Miyazaki")
    RAG.index(
        collection=[full_document],
        index_name="Miyazaki",
        max_document_length=180,
        split_documents=True,
    )
    return RAG

def search(RAG:RAGPretrainedModel, qyery:str = "") ->List[Dict]:
    results = RAG.search(query="What animation studio did Miyazaki found?", k=3)

def get_retriever(rag:RAGPretrainedModel) -> CustomRetriever:
    return CustomRetriever(rag=rag)

def get_chain(rag:RAGPretrainedModel):
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}"""
    )

    llm = ChatOpenAI()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(rag.as_langchain_retriever(k=3), document_chain)
    return retrieval_chain


def stream_ask(retrieval_chain, input:str="What animation studio did Miyazaki found?"):
    for s in retrieval_chain.stream({"input": input}):
        print(s.get("answer", ""), end="")

def load_rag(index_path:str=".ragatouille/colbert/indexes/Miyazaki/") -> RAGPretrainedModel:
    return RAGPretrainedModel.from_index(index_path)


rag = get_rag()
rag_index(rag)
rag = load_rag()
rag.serach('')
