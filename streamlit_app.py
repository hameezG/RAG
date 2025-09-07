import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# ✅ Load secrets for Streamlit Cloud
PINECONE_KEY = st.secrets.get("PINECONE_API_KEY", "")
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
os.environ["PINECONE_API_KEY"] = PINECONE_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# ✅ Use HuggingFace for embedding
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# ✅ Connect to existing Pinecone index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="gcse-bge-index",
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ Set up the OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ✅ Build QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ✅ Wrap chain as a LangChain tool
kb_tool = Tool(
    name="Knowledge Base",
    func=qa_chain.run,
    description="Answer questions from GCSE books"
)

# ✅ Memory to track conversation
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=3, return_messages=True
)

# ✅ Initialize LangChain agent
agent_executor = initialize_agent(
    tools=[kb_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# ✅ Streamlit UI
st.set_page_config(page_title="GCSE RAG Chatbot", page_icon="📘")
st.title("📘 GCSE RAG Chatbot")
query = st.text_input("Ask a question from your GCSE books:")

# ✅ Handle user query
if query:
    st_callback = StreamlitCallbackHandler(st.container())
    response = agent_executor.run(query, callbacks=[st_callback])
