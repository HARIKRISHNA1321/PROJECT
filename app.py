import os
import streamlit as st

# Updated imports to use langchain_classic
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- 1. SESSION STATE SETUP ---
# This allows Streamlit to remember data across page reloads
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "employee_status" not in st.session_state:
    st.session_state.employee_status = "Pending: Missing Acceptance Letter"

# Navigation helper functions
def go_to_details():
    st.session_state.current_page = "details"

def go_to_dashboard():
    st.session_state.current_page = "dashboard"

# --- 2. PAGE: LOGIN / SIGNUP ---
def login_page():
    st.title("Company Portal - Login")
    st.write("Welcome to the Onboarding System.")
    
    with st.form("login_form"):
        username = st.text_input("Username / Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In / Sign Up")
        
        if submitted:
            if username and password:
                st.session_state.user_info['username'] = username
                go_to_details()
            else:
                st.error("Please enter your credentials.")

# --- 3. PAGE: PERSONAL DETAILS ---
def details_page():
    st.title("Complete Your Profile")
    st.write("Please provide your onboarding details so we can set up your workspace.")
    
    with st.form("details_form"):
        full_name = st.text_input("Full Name")
        department = st.selectbox("Department", ["Engineering", "HR", "Marketing", "Sales", "Finance"])
        role = st.text_input("Job Role")
        submitted = st.form_submit_button("Save Details & Continue")
        
        if submitted:
            if full_name and role:
                st.session_state.user_info['full_name'] = full_name
                st.session_state.user_info['department'] = department
                st.session_state.user_info['role'] = role
                go_to_dashboard()
            else:
                st.error("Please fill out all fields.")

# --- 4. PAGE: DASHBOARD & AI AGENT ---
def dashboard_page():
    st.title("HR-ONBOARDING AGENTIC AI")
    
    # UI Sidebar: Display the requested Employee Status and Details
    with st.sidebar:
        st.header("Employee Profile")
        st.write(f"**Name:** {st.session_state.user_info.get('full_name', 'N/A')}")
        st.write(f"**Role:** {st.session_state.user_info.get('role', 'N/A')}")
        st.write(f"**Dept:** {st.session_state.user_info.get('department', 'N/A')}")
        
        st.divider()
        
        st.subheader("Onboarding Status")
        st.warning(st.session_state.employee_status)
    
    # Load API Keys securely
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

    # Setup Pinecone & Langchain
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = PineconeVectorStore(index_name="gemini-rag-3072-working", embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = ChatPromptTemplate.from_template("Answer based on: {context} \n\nQuestion: {input}")
    
    # Using the classic chains here
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    
    # Chat Interface
    query = st.text_input("Ask a question about your onboarding policies:")
    if query:
        with st.spinner("Searching company database..."):
            response = chain.invoke({"input": query})
            st.success(response["answer"])

# --- 5. MAIN ROUTER LOGIC ---
if st.session_state.current_page == "login":
    login_page()
elif st.session_state.current_page == "details":
    details_page()
elif st.session_state.current_page == "dashboard":
    dashboard_page()
