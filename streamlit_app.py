import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 🔑 Load secrets (set in Streamlit Cloud or .streamlit/secrets.toml)
openai_key = st.secrets["OPENAI_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["PINECONE_API_KEY"] = pinecone_key

# embeddings + Pinecone
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="gcse-bge-index",
    embedding=embeddings
)

# retriever + LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Streamlit UI
st.title("📘 GCSE RAG Chatbot")
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
    st.write("### 📖 Answer")
    st.write(response)
