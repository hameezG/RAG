# streamlit_app.py
import streamlit as st
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# --- Streamlit page setup ---
st.set_page_config(page_title="GCSE Book Chatbot", page_icon="📚", layout="wide")
st.title("📚 GCSE Book Chatbot")
st.write("Ask questions from your GCSE PDFs and get accurate answers!")

# --- API Keys from Streamlit secrets ---
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Pinecone setup ---
INDEX_NAME = "gcse-bge-index"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# Connect to vector store
vectorstore = PineconeVectorStore(
    index=pc.Index(INDEX_NAME),
    embedding=embeddings,
    text_key="text"
)

# --- LLM setup ---
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# --- Chat interface ---
if "history" not in st.session_state:
    st.session_state.history = []

user_question = st.text_input("Enter your question:")

if st.button("Ask") and user_question:
    with st.spinner("Generating answer..."):
        result = qa_chain.run(user_question)
        st.session_state.history.append({"question": user_question, "answer": result})

# Display chat history
for chat in st.session_state.history[::-1]:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")
