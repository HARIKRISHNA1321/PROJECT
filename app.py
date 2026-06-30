import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="PDF Assistant", page_icon="📚")
st.title("📚 Your Agentic PDF Assistant")
st.write("Ask questions based on your uploaded documents!")

# 2. Securely load API keys from Streamlit Secrets
# We set them as environment variables so LangChain can find them automatically
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# 3. Initialize the RAG Chain (Cached so it doesn't reload on every chat message)
@st.cache_resource
def init_rag_chain():
    # Set up the Embeddings and Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    index_name = "gemini-rag-3072"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    
    # Configure the Retriever (How many chunks to fetch)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Set up the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    # Create the System Prompt
    system_prompt = (
        "You are a highly intelligent and helpful AI assistant. "
        "Use the following pieces of retrieved context from a PDF document to answer the user's question. "
        "If you don't know the answer based on the context, simply say that you don't know. "
        "Keep your answers clear, concise, and accurate.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Combine everything into a single answering chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Start up the AI engine
rag_chain = init_rag_chain()

# 4. Set up Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Chat Input box and Response Logic
if user_input := st.chat_input("Ask a question about your PDF..."):
    # Show user message
    st.chat_message("user").markdown(user_input)
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate and show AI response
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                # Send the question to the LangChain RAG system
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                
                # Print the answer to the screen
                st.markdown(answer)
                
                # Save the answer to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")