import time
import requests
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from langchain_community.embeddings import HuggingFaceHubEmbeddings

from src.claims.utils import get_constants

config = get_constants().EMBEDDERS.get("fb_aws_embedder")    
default_embedder = HuggingFaceHubEmbeddings(
    model = config.api_url,
    huggingfacehub_api_token=config.api_token
)

class CustomEmbedder(EmbeddingFunction):
    def __init__(self, API_URL, API_TOKEN):
        self.API_URL = API_URL
        self.API_TOKEN = API_TOKEN
    def __call__(self, input: Documents) -> Embeddings:
        reponse = requests.post(
            self.API_URL,
            json={"inputs": input},
            headers={"Authorization": f"Bearer {self.API_TOKEN}"}
        ).json()
        return reponse
    def embed_documents(self, documents):
        return self(documents)
    def embed_query(self, document):
        #time.sleep(30)
        return self.embed_documents([document])[0]

