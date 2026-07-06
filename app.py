import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client

# Langchain Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- 1. SETUP CLOUD DATABASE & SESSION ---
@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_connection()

if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "current_user" not in st.session_state:
    st.session_state.current_user = {}

# Navigation Helper
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
                st.rerun()
            else:
                navigate_to("details")
                st.rerun()

# --- 3. PAGE: APPLICATION / DETAILS ---
def details_page():
    st.title("New Employee Application")
    st.write("Please provide your details for HR approval.")
    
    with st.form("details_form"):
        full_name = st.text_input("Full Name")
        department = st.selectbox("Department", ["CSE", "CSE(AIML)", "ECE", "ME", "EEE"])
        submitted = st.form_submit_button("Submit Application to HR")
        
        if submitted:
            if full_name:
                # Save to user session
                st.session_state.current_user['full_name'] = full_name
                st.session_state.current_user['department'] = department
                st.session_state.current_user['status'] = "Pending HR Approval"
                
                # Insert directly into Supabase 'applications' table
                supabase.table("applications").insert({
                    "name": full_name,
                    "role": st.session_state.current_user['role'],
                    "department": department,
                    "status": "Pending"
                }).execute()
                
                navigate_to("employee_dashboard")
                st.rerun()
            else:
                st.error("Please fill out all fields.")

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
        
        if st.button("Log Out / Switch User"):
            st.session_state.current_user = {} 
            navigate_to("login")
            st.rerun()
            
        st.divider()
        
        st.subheader("Submit Leave Letter")
        with st.form("leave_form", clear_on_submit=True):
            leave_dates = st.text_input("Dates (e.g., Oct 12 - Oct 14)")
            leave_reason = st.text_area("Reason")
            submit_leave = st.form_submit_button("Send to HR")
            
            if submit_leave and leave_dates:
                # Insert directly into Supabase 'leaves' table
                supabase.table("leaves").insert({
                    "name": user.get('full_name', 'N/A'),
                    "department": user.get('department', 'N/A'),
                    "dates": leave_dates,
                    "reason": leave_reason,
                    "status": "Pending"
                }).execute()
                st.success("Leave letter sent to HR.")

    # Load API Keys
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

    st.subheader("Onboarding AI Assistant")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vectorstore = PineconeVectorStore(index_name="gemini-rag-3072-working", embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
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
    
    with st.sidebar:
        if st.button("Log Out / Switch User"):
            st.session_state.current_user = {}
            navigate_to("login")
            st.rerun()

    # Fetch live data from Supabase
    db_applications = supabase.table("applications").select("*").eq("status", "Pending").execute().data
    db_leaves = supabase.table("leaves").select("*").eq("status", "Pending").execute().data
    db_employees = supabase.table("employees").select("*").execute().data

    # 5A. New Applications
    st.header("1. New Applications")
    if not db_applications:
        st.write("No new applications.")
    else:
        for app in db_applications:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.write(f"**{app['name']}** - {app['role']} ({app['department']})")
            
            # Note: We now use the unique database 'id' for button keys!
            if col2.button("Approve", key=f"app_approve_{app['id']}"):
                # Add to employees table
                supabase.table("employees").insert({
                    "full_name": app['name'],
                    "role": app['role'],
                    "department": app['department'],
                    "status": "Active"
                }).execute()
                # Delete from applications table
                supabase.table("applications").delete().eq("id", app['id']).execute()
                st.rerun()
                
            if col3.button("Reject", key=f"app_reject_{app['id']}"):
                # Delete from applications table
                supabase.table("applications").delete().eq("id", app['id']).execute()
                st.rerun()

    st.divider()

    # 5B. Leave Requests
    st.header("2. Pending Leave Letters")
    if not db_leaves:
        st.write("No pending leave requests.")
    else:
        for leave in db_leaves:
            st.write(f"**{leave['name']}** ({leave['department']}) | Dates: {leave['dates']}")
            st.write(f"Reason: {leave['reason']}")
            col1, col2 = st.columns([1, 8])
            
            if col1.button("Approve Leave", key=f"leave_{leave['id']}"):
                # Update status in the leaves table
                supabase.table("leaves").update({"status": "Approved"}).eq("id", leave['id']).execute()
                st.rerun()

    st.divider()

    # 5C. Employee Directory
    st.header("3. Employee Directory")
    if db_employees:
        # Convert the Supabase JSON response to a Pandas DataFrame
        df = pd.DataFrame(db_employees)
        # Drop the database ID column so it looks cleaner on the dashboard
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
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
