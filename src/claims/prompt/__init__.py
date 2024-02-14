from langchain_core.prompts import PromptTemplate

template = """Instructions: Use only the following context to answer the question. Answer should be short and precise. Do not provide any additional questions and answers in the response

Context: {context}
Question: {question}
"""
def get_prompt():
    return PromptTemplate(template=template, input_variables=["context", "question"])
