import os
import time
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader = DirectoryLoader('folder-with-txt-files/', glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

os.environ['OPENAI_API_KEY'] = "your-openai-api-key"

text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

