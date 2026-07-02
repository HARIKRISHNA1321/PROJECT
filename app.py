import os
import streamlit as st

# Updated imports to use langchain_classic
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# ... the rest of your app code ...
# 1. Setup
st.title("HR-ONBOARDING AGENTIC AI")
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# 2. Connect to your Pinecone index
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = PineconeVectorStore(index_name="gemini-rag-3072-working", embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Setup LLM and Chain
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
prompt = ChatPromptTemplate.from_template("Answer based on: {context} \n\nQuestion: {input}")
chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# 4. Interface
query = st.text_input("Ask a question:")
if query:
    response = chain.invoke({"input": query})
    st.write(response["answer"])
