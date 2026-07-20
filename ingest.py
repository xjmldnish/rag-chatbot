from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()  # reads your API key from the .env file

# 1. Read every PDF inside the docs folder
loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Cut the text into overlapping chunks so no idea gets sliced in half
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. Convert chunks to embeddings and save them in a local database folder
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print("Done! Knowledge base saved to ./chroma_db")