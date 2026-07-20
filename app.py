import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

@st.cache_resource  # load models once, not on every click
def load_components():
    emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vs = Chroma(persist_directory="./chroma_db", embedding_function=emb)
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this context. If the answer is not in it, "
        "say you don't have that information.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return retriever, llm, prompt

retriever, llm, prompt = load_components()

st.title("🧠 AI Research Paper Assistant")
st.caption("Answers are grounded in the source PDFs (Transformers, RAG, LoRA, RLHF, Chain-of-Thought) — sources shown below each answer.")

if question := st.chat_input("Ask a question..."):
    st.chat_message("user").write(question)

    docs = retriever.invoke(question)                     # find relevant chunks
    context = "\n\n".join(d.page_content for d in docs)   # merge them into one text
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )

    with st.chat_message("assistant"):
        st.write(answer)
        with st.expander("📚 Sources"):
            for d in docs:
                name = d.metadata.get("source", "unknown")
                page = d.metadata.get("page", "?")
                st.markdown(f"**{name}** — page {page}")
                st.caption(d.page_content[:200] + "...")