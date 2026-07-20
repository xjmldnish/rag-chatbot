from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Open the knowledge base you built yesterday
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # fetch top 4 chunks

# The AI model that writes the answers
# (check ai.google.dev for the current flash model name if this errors)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# The instructions we wrap around every question
prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information in my documents."
Do not make anything up.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# Chain the steps together: retrieve -> fill prompt -> ask Gemini -> clean output
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = input("Ask a question about your documents: ")
    print("\n" + rag_chain.invoke(question))