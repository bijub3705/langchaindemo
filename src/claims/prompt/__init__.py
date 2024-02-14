from langchain_core.prompts import PromptTemplate

template = """You are a helpful AI assistant. 
    Use the following context to answer the question at the end. Stop when you've answered the question. Do not generate any more than that.
    Respond "Please elaborate more" when you are unable to find an answer.
        
    Consider the following information to answer questions:

    Context: {context}
    Question: {question}
"""
def get_prompt():
    return PromptTemplate(template=template, input_variables=["context", "question"])
