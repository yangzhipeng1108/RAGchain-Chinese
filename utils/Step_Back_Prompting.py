import os
import openai
os.environ['OPENAI_API_KEY'] = str("xxxxxxxxxxxxxxxxxxxxxxxxxxx")
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

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

    def decompose(self, query: str) -> List[str]:
        """
        decompose query to little piece of questions.
        :param query: str, query to decompose.
        :return: List[str], list of decomposed query. Return empty list if query is not decomposable.
        """
        from langchain import hub

        response_prompt = hub.pull("langchain-ai/stepback-answer")

        chain =  response_prompt | ChatOpenAI(temperature=0) | StrOutputParser()

        answer = chain.invoke({"question": query})

        if answer.lower().strip() == "the question needs no decomposition.":
            return []
        try:
            questions = [l for l in answer.splitlines() if l != ""]
            questions = [q.split(':')[1].strip() for q in questions]
            return questions
        except:
            return []



def funct():

    examples = [
        {
            "input": "Could the members of The Police perform lawful arrests?",
            "output": "what can the members of The Police do?"
        },
        {
            "input": "Jan Sindel’s was born in what country?",
            "output": "what is Jan Sindel’s personal history?"
        },
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:"""),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ])


    question_gen = prompt | ChatOpenAI(temperature=0) | StrOutputParser()

    question = "was chatgpt around while trump was president?"

    question_gen.invoke({"question": question})

    from langchain.utilities import DuckDuckGoSearchAPIWrapper

    search = DuckDuckGoSearchAPIWrapper(max_results=4)

    def retriever(query):
        return search.run(query)

    retriever(question)

    retriever(question_gen.invoke({"question": question}))


    from langchain import hub

    response_prompt = hub.pull("langchain-ai/stepback-answer")


    chain = {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x['question']) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": question_gen | retriever,
        # Pass on the question
        "question": lambda x: x["question"]
    } | response_prompt | ChatOpenAI(temperature=0) | StrOutputParser()


    chain.invoke({"question": question})


    response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.
    
    {normal_context}
    
    Original Question: {question}
    Answer:"""
    response_prompt = ChatPromptTemplate.from_template(response_prompt_template)


    chain = {
        # Retrieve context using the normal question (only the first 3 results)
        "normal_context": RunnableLambda(lambda x: x['question']) | retriever,
        # Pass on the question
        "question": lambda x: x["question"]
    } | response_prompt | ChatOpenAI(temperature=0) | StrOutputParser()


    chain.invoke({"question": question})