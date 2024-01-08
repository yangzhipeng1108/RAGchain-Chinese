from typing import List

from RAGchain.reranker.base import BaseReranker
from RAGchain.schema import Passage
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BGEReranker(BaseReranker):
    """
    BM25Reranker class for reranker based on BM25.
    You can rerank the passages with BM25 scores .
    """

    def __init__(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
        self.model.eval()

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        contents: List[str] = [passage.content for passage in passages]

        scores = []
        for i in contents:
            pairs = [query] + [i]
            with torch.no_grad():
                inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                score = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores.append(score)

        sorted_pairs = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

        sorted_passages = [passage for passage, _ in sorted_pairs]

        return sorted_passages
