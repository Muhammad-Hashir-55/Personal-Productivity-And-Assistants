import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import gdown



load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


documents = [
    Document(page_content="Meeting notes: Discuss project X deliverables"),
    Document(page_content="Reminder: submit report by friday"),
    Document(page_content="Upcoming event: tech conference next week")
]

embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY)

text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
documents = text_splitter.split_documents(documents)

# Create and save the vector store
vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH
)

# Persist the vector store to disk
print("Documents ingested successfully")