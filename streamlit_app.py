import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 🔑 Set API keys
os.environ["PINECONE_API_KEY"] = "pcsk_5FNTdu_5T7gq63GSCeDmHfEhQEsphroiwH4oyGMaY3pY9fAXJymVLzxTJESBUtTXxd3b4t"
os.environ["OPENAI_API_KEY"] = "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"  # if using OpenAI

# 1. Init embeddings (must match what you used before)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# 2. Connect to Pinecone index
index_name = "gcse-bge-index"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 3. Setup Retriever + LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # swap to gpt-3.5-turbo if cheaper

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 4. Streamlit UI
st.set_page_config(page_title="GCSE Chatbot", page_icon="📘")
st.title("📘 GCSE RAG Chatbot")
st.write("Ask me anything from your GCSE books and I’ll find the answer.")

# Chat input
query = st.text_input("Enter your question:")

if query:
    with st.spinner("Searching your GCSE books..."):
        response = qa_chain.run(query)
    st.markdown("### 📖 Answer")
    st.write(response)
