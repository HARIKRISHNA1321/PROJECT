import os
import streamlit as st
import pandas as pd

# Langchain Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- 1. MOCK DATABASE (SESSION STATE) ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "current_user" not in st.session_state:
    st.session_state.current_user = {}

# Storing lists of dictionaries to act as our database tables
if "db_employees" not in st.session_state:
    st.session_state.db_employees = []
if "db_applications" not in st.session_state:
    st.session_state.db_applications = []
if "db_leaves" not in st.session_state:
    st.session_state.db_leaves = []

# Navigation Helpers
def navigate_to(page):
    st.session_state.current_page = page

# --- 2. PAGE: LOGIN ---
def login_page():
    st.title("Institution Portal - Login")
    st.write("Welcome to the Faculty & Staff Onboarding System.")
    
    with st.form("login_form"):
        username = st.text_input("Username / Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Log in as:", ["Teacher", "Officer", "HR"])
        submitted = st.form_submit_button("Log In / Sign Up")
        
        if submitted and username:
            st.session_state.current_user = {'username': username, 'role': role}
            
            if role == "HR":
                navigate_to("hr_dashboard")
            else:
                navigate_to("details")

# --- 3. PAGE: APPLICATION / DETAILS ---
def details_page():
    st.title("New Employee Application")
    st.write("Please provide your details for HR approval.")
    
    with st.form("details_form"):
        full_name = st.text_input("Full Name")
        department = st.selectbox("Department", ["CSE", "CSE(AIML)", "ECE", "ME", "EEE"])
        submitted = st.form_submit_button("Submit Application to HR")
        
        if submitted and full_name:
            # Save to user session
            st.session_state.current_user['full_name'] = full_name
            st.session_state.current_user['department'] = department
            st.session_state.current_user['status'] = "Pending HR Approval"
            
            # Add to HR's application database
            st.session_state.db_applications.append({
                "Name": full_name,
                "Role": st.session_state.current_user['role'],
                "Department": department,
                "Status": "Pending"
            })
            
            navigate_to("employee_dashboard")

# --- 4. PAGE: EMPLOYEE DASHBOARD & AI ---
def employee_dashboard():
    user = st.session_state.current_user
    st.title("Staff Dashboard & Support")
    
    # Sidebar: Profile & Leave Request
    with st.sidebar:
        st.header("Profile")
        st.write(f"**Name:** {user.get('full_name', 'N/A')}")
        st.write(f"**Role:** {user.get('role', 'N/A')}")
        st.write(f"**Dept:** {user.get('department', 'N/A')}")
        st.warning(f"Status: {user.get('status', 'Unknown')}")
        
        st.divider()
        
        st.subheader("Submit Leave Letter")
        with st.form("leave_form", clear_on_submit=True):
            leave_dates = st.text_input("Dates (e.g., Oct 12 - Oct 14)")
            leave_reason = st.text_area("Reason")
            submit_leave = st.form_submit_button("Send to HR")
            
            if submit_leave and leave_dates:
                st.session_state.db_leaves.append({
                    "Name": user.get('full_name', 'N/A'),
                    "Department": user.get('department', 'N/A'),
                    "Dates": leave_dates,
                    "Reason": leave_reason,
                    "Status": "Pending"
                })
                st.success("Leave letter sent to HR.")

    # Load API Keys
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

    st.subheader("Onboarding AI Assistant")
    
    try:
        # Setup Models as requested
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = PineconeVectorStore(index_name="gemini-rag-3072-working", embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
        prompt = ChatPromptTemplate.from_template("Answer based on: {context} \n\nQuestion: {input}")
        
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        
        query = st.text_input("Ask about institution policies or onboarding:")
        if query:
            with st.spinner("Searching records..."):
                response = chain.invoke({"input": query})
                st.info(response["answer"])
    except Exception as e:
        st.error(f"AI Connection Error: {e}")

# --- 5. PAGE: HR ADMIN DASHBOARD ---
def hr_dashboard():
    st.title("HR Administration Dashboard")
    st.write("Manage applications, personnel, and leave requests.")
    
    # 5A. New Applications
    st.header("1. New Applications")
    if not st.session_state.db_applications:
        st.write("No new applications.")
    else:
        for idx, app in enumerate(st.session_state.db_applications):
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"**{app['Name']}** - {app['Role']} ({app['Department']})")
            if col2.button("Approve", key=f"app_approve_{idx}"):
                # Move to employee DB and remove from apps
                st.session_state.db_employees.append(app)
                st.session_state.db_applications.pop(idx)
                st.rerun()
            if col3.button("Reject", key=f"app_reject_{idx}"):
                st.session_state.db_applications.pop(idx)
                st.rerun()

    st.divider()

    # 5B. Leave Requests
    st.header("2. Pending Leave Letters")
    if not st.session_state.db_leaves:
        st.write("No pending leave requests.")
    else:
        for idx, leave in enumerate(st.session_state.db_leaves):
            if leave["Status"] == "Pending":
                st.write(f"**{leave['Name']}** ({leave['Department']}) | Dates: {leave['Dates']}")
                st.write(f"Reason: {leave['Reason']}")
                col1, col2 = st.columns([1, 8])
                if col1.button("Approve Leave", key=f"leave_{idx}"):
                    st.session_state.db_leaves[idx]["Status"] = "Approved"
                    st.rerun()

    st.divider()

    # 5C. Employee Directory
    st.header("3. Employee Directory")
    if st.session_state.db_employees:
        # Using Pandas to display a clean table in Streamlit
        df = pd.DataFrame(st.session_state.db_employees)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No active employees in the system yet.")

# --- 6. MAIN ROUTER LOGIC ---
if st.session_state.current_page == "login":
    login_page()
elif st.session_state.current_page == "details":
    details_page()
elif st.session_state.current_page == "employee_dashboard":
    employee_dashboard()
elif st.session_state.current_page == "hr_dashboard":
    hr_dashboard()
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
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
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
