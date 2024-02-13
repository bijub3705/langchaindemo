from langchain_community.vectorstores.chroma import Chroma
from src.claims.vector_db.custom_embedder import CustomEmbedder
from src.claims.utils import get_constants
from src.claims.vector_db.custom_embedder import default_embedder

class ChromaDB:
    def __init__(self):
        self.embedder_config = get_constants().EMBEDDERS.get("fb_aws_embedder")
        self.embedder = CustomEmbedder(API_URL=self.embedder_config.api_url, API_TOKEN=self.embedder_config.api_token)
        #self.embedder = default_embedder
        self.db = Chroma(
            persist_directory="./chromadb",
            embedding_function=self.embedder
        )

    def add_data(self, data):
        self.db.add_documents(documents=data)
    
    def retrieve_data(self, query):
        return self.db.similarity_search(query=query)
    
    def retriever(self):
        return self.db.as_retriever(search_kwargs={"k": 1})