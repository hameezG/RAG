# app.py - Streamlit app for GCSE Chatbot using Pinecone and LLM

import os
import streamlit as st
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# Set API keys (replace with your actual keys or use environment variables)
# For security, set these in Streamlit's secrets or as env vars in your deployment
os.environ["PINECONE_API_KEY"] = "pcsk_5FNTdu_5T7gq63GSCeDmHfEhQEsphroiwH4oyGMaY3pY9fAXJymVLzxTJESBUtTXxd3b4t"
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your OpenAI API key

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Index name (must match your existing Pinecone index)
index_name = "gcse-bge-index"

# Load embedding model (same as used for vectorization)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# Connect to existing Pinecone vector store
vectorstore = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings,
    text_key="text",  # Must match what you used during upload
    namespace=""  # Empty namespace as per your code; change if you used a different one
)

# Initialize LLM (using OpenAI's GPT-3.5-turbo for accurate, factual responses)
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0  # Low temperature for deterministic, accurate answers
)

# Set up RetrievalQA chain
# Chain type "stuff" is simple and works well for single-query retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 chunks; adjust as needed
    return_source_documents=True  # Optional: Returns sources for transparency
)

# Streamlit app layout
st.title("GCSE Chatbot")
st.markdown("Ask questions about your GCSE books. The bot uses vector search from Pinecone for accurate answers.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question here (e.g., 'What is a learner’s overall qualification grade for OCR GCSE?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]
                # Optional: You can access result["source_documents"] here if you want to display sources
            except Exception as e:
                response = f"An error occurred: {str(e)}. Please check your API keys and Pinecone connection."
        
        st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})