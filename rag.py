from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

chunk = text_splitter.split_documents(documents=docs)

print

embedding_model = OpenAIEmbeddings(
    model="text-embeddings-3-large"
)

vector_store = QdrantVectorStore.from_documents(
    documents=docs,
    embedding=embedding_model,
    url=qdrant_url,
    api_key=qdrant_api_key,
    collection="rag-leraning"
)

print("indexing is done")