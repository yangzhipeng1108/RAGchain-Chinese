"""
This code is inspired by Visconde paper and its github repo.
@inproceedings{10.1007/978-3-031-28238-6_44,
author = {Pereira, Jayr and Fidalgo, Robson and Lotufo, Roberto and Nogueira, Rodrigo},
title = {Visconde: Multi-Document QA With&nbsp;GPT-3 And&nbsp;Neural Reranking},
year = {2023},
isbn = {978-3-031-28237-9},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-28238-6_44},
doi = {10.1007/978-3-031-28238-6_44},
booktitle = {Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2–6, 2023, Proceedings, Part II},
pages = {534–543},
numpages = {10},
location = {Dublin, Ireland}
}
"""
import os
import openai
import random
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from typing import List


# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")  # Alternative: Use environment variable
if openai.api_key is None:
    raise Exception("No OpenAI API key found. Please set it as an environment variable or in main.py")

# Function to generate queries using OpenAI's ChatGPT



class QueryDecomposition:
    """
    Query Decomposition class.
    You can decompose a multi-hop questions to multiple single-hop questions using LLM.
    The default decomposition prompt is from Visconde paper, and its prompt is few-shot prompts from strategyQA dataset.
    """


    def __init__(self, llm: BaseLLM):
        """
        :param llm: BaseLLM, language model to use. Query Decomposition not supports chat model. Only supports completion LLMs.
        """
        self.llm = llm

    def generate_queries_chatgpt(self, original_query) -> List[str]:

        if self.llm is None:

            self.llm = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
                    {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
                    {"role": "user", "content": "OUTPUT (4 queries):"}
                ]
            )

        generated_queries = self.llm.choices[0]["message"]["content"].strip().split("\n")
        return generated_queries
