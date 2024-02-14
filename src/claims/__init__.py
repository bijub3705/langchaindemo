from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.claims.llm import get_llm
from src.claims.vector_db import get_chroma_db
from src.claims.prompt import get_prompt

class Claim_RAG:
    def __init__(self, name):
        self.name = name
        self.model = get_llm("summarize-model")
        self.vector_db = get_chroma_db()
        self.retriever = self.vector_db.retriever()
        self.prompt = get_prompt()
     
    def respond_to_query(self, query):
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        return chain.invoke(query)