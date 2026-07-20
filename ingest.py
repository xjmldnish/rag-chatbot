import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

loader = PyPDFDirectoryLoader("docs/")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Free tier allows 100 embedding requests/minute -> send batches of 90,
# then wait 65 seconds before the next batch.
BATCH_SIZE = 90
vectorstore = None

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            batch, embeddings, persist_directory="./chroma_db"
        )
    else:
        vectorstore.add_documents(batch)
    done = min(i + BATCH_SIZE, len(chunks))
    print(f"Embedded {done}/{len(chunks)} chunks")
    if done < len(chunks):
        print("Waiting 65s for the rate limit window...")
        time.sleep(65)

print("Done! Knowledge base saved to ./chroma_db")