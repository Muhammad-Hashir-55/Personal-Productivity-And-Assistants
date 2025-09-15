import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import openaiembeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import document
import gdown



load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "db")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


documents = [
    Document(page_content="Meeting notes: Discuss project X deliverables")
    Document(page_content="Reminder: submit report by friday")
    Document(page_content="Upcoming event: tech conference next week")
]