import streamlit as st
from langchain.schema import(SystemMessage, HumanMessage, AIMessage)
from src.claims.llm import get_llm
from io import StringIO
from src.claims.vector_db import get_chroma_db

llm = get_llm("fb_aws_llm")

def init_page() -> None:
  st.set_page_config(page_title="Claim Chatbot")
  st.header("Claim Chatbot")
  st.sidebar.title("Options")


def init_messages() -> None:
  clear_button = st.sidebar.button("Clear Conversation", key="clear")
  if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages = [
      SystemMessage(
        content="""You are a helpful AI assistant. 
        Use the following context to answer the question at the end. Stop when you've answered the question. Do not generate any more than that.
        Respond "Please elaborate more" when you are unable to find an answer.
        
        Consider the following information to answer questions: \n
        """
      )
    ]
     

def get_answer(prompt) -> str:
  return llm._call(prompt)

def get_current_prompt(user_input)-> None:
    chunked_vector_data = get_chroma_db().retrieve_data(user_input)
    #print(chunked_vector_data)
    prompt_system_message = f"""
    The following paragraph is a general information to consider:
    {chunked_vector_data}
    """
    st.session_state.messages.append(SystemMessage(content=prompt_system_message))
    st.session_state.messages.append(HumanMessage(content=user_input))
    prompt = ""
    messages = st.session_state.get("messages", [])
    for message in messages: 
        if isinstance(message, SystemMessage):
            prompt += message.content + "\n"
        elif isinstance(message, AIMessage):
            prompt += message.content + "\n"
        elif isinstance(message, HumanMessage):
            prompt += "Question :" +message.content + "? \n"
    
    print(prompt)
    return prompt

def main() -> None:
  init_page()
  init_messages()

  if user_input := st.chat_input("Ask your question!"):
    with st.spinner("AI assistant is finding the answer for you ..."):
      updated_prompt = get_current_prompt(user_input)
      answer = get_answer(updated_prompt)
      #print(answer)
    st.session_state.messages.append(AIMessage(content=answer))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
             with st.chat_message("user"):
                st.markdown(message.content)


if __name__ == "__main__":
  main()