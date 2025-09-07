import streamlit as st
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# Access API keys from Streamlit secrets
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
huggingface_api_key = st.secrets["HUGGINGFACE_API_KEY"]

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Index name (matches your existing Pinecone index)
index_name = "gcse-bge-index"

# Load embedding model (same as used in Colab for vectorization)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# Connect to existing Pinecone vector store
vectorstore = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=embeddings,
    text_key="text",
    namespace=""
)

# Initialize LLM (Hugging Face model)
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mixtralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token=huggingface_api_key,
    temperature=0.0,
    max_new_tokens=512
)

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# Streamlit app layout
st.title("GCSE Chatbot")
st.markdown("Ask questions about your GCSE books. The bot uses vector search from Pinecone for accurate answers.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your question (e.g., 'What is a learner’s overall qualification grade for OCR GCSE?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]
            except Exception as e:
                response = f"An error occurred: {str(e)}. Please check your API keys and Pinecone connection."
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})