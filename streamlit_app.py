import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Load secrets for production (Streamlit Cloud)
PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = PINECONE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="gcse-bge-index", embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Set up LLM and QA chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, return_source_documents=True
)

# Wrap as an agent for Streamlit
kb_tool = Tool(name="Knowledge Base", func=qa_chain.run,
               description="Answer questions using GCSE content")
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)
agent_executor = initialize_agent(
    tools=[kb_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# UI setup
st.set_page_config(page_title="GCSE RAG Chatbot", page_icon="📘")
st.title("📘 GCSE RAG Chatbot")
query = st.text_input("Ask a question from your GCSE books:")

if query:
    st_callback = StreamlitCallbackHandler(st.container())
    response = agent_executor.run(query, callbacks=[st_callback])
    # The callback handler will show chain steps; final answer shows automatically.
