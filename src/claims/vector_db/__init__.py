from src.claims.vector_db.chroma_db import ChromaDB
# Initialize the Chroma database
chroma_db = ChromaDB()

def get_chroma_db():
    return chroma_db
# Add data
def add_data_to_chroma_db(data):
    chroma_db.add_data(data)
# Retrieve data
def retrieve_data_from_chroma_db(query_text):
    retrieved_data = chroma_db.retrieve_data(query_text)
    return retrieved_data