from typing import Optional, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage
from langchain.retrievers import ParentDocumentRetriever

class RecursiveTextSplitter(BaseTextSplitter):
    """
    Split a document into passages by recursively splitting on a list of separators.
    You can specify a window_size and overlap_size to split the document into overlapping passages.
    """
    def __init__(self,vectorstore, separators: Optional[List[str]] = None,
                 keep_separator: bool = True,
                 *args, **kwargs):
        """
        :param separators: A list of strings to split on. Default is None.
        :param keep_separator: Whether to keep the separator in the passage. Default is True.
        :param kwargs: Additional arguments to pass to the langchain RecursiveCharacterTextSplitter.
        """
        # 创建主文档分割器
        parent_splitter = RecursiveCharacterTextSplitter(separators, keep_separator, chunk_size=1000, **kwargs)

        # 创建子文档分割器
        child_splitter = RecursiveCharacterTextSplitter(separators, keep_separator, chunk_size=400, **kwargs)

        # 创建向量数据库对象
        # 创建父文档检索器
        self.splitter  = ParentDocumentRetriever(
                vectorstore=vectorstore,
                # docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={"k": 1}
            )

    def split_document(self, document: Document) -> List[Passage]:
        """
        Split a document.
        """
        split_documents = self.splitter.add_documents([document])
        passages = self.docs_to_passages(split_documents)

        return passages
