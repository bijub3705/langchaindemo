from langchain_core.prompts import PromptTemplate

template = """Instructions: Use only the following context to answer the question.

Context: {context}
Question: {question}
"""
def get_prompt():
    return PromptTemplate(template=template, input_variables=["context", "question"])
