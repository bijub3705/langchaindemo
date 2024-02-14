import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import AirbyteJSONLoader
from langchain.text_splitter import CharacterTextSplitter
from src.claims.vector_db import get_chroma_db
from src.claims.document_loader.getclaim import getClaimData
class DocumentLoader:
    def __init__(self):
        self.doc_folder_path = "docs"
        self.chroma_db = get_chroma_db()

    def load(self):
        documents = []
        root_path = self.doc_folder_path
        for file in os.listdir(root_path):
            doc_path = root_path+"/"+file
            if file.endswith(".pdf"):
                document=PyPDFLoader(doc_path).load()
            elif file.endswith(".txt"):
                document=TextLoader(doc_path).load()
            elif file.endswith(".json"):
                document=AirbyteJSONLoader(doc_path).load()
            elif file.endswith(".docx") or file.endswith(".doc"):
                document=Docx2txtLoader(doc_path).load()
            documents.extend(document)
        if len(documents) > 0:
            self.load_chunk_persist_data(documents)
    def load_claim_data(self):
        documents=getClaimData("docs/context_help.csv")
        self.chroma_db.add_data(documents)
        
    def load_chunk_persist_data(self,documents):
        document_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=10)
        document_chunks = document_splitter.split_documents(documents)
        self.chroma_db.add_data(document_chunks)
       
