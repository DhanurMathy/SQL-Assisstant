import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sql_components import setup_database
import re
import plotly.express as px
import pandas as pd
import json

# Initialize Database Connection
db = setup_database()
if db is None:
    st.error("Failed to connect to the database")

# AI Model for Query Processing
llm = ChatGroq(model="qwen-2.5-32b", temperature=0)

def sql_assistant_page():
    """SQL Assistant Interface"""
    st.title("Multilingual SQL Assistant")

    # Language Selection
    languages = ["English", "Spanish", "French", "German", "Chinese", "Tamil"]
    selected_language = st.sidebar.selectbox("Select Output Language", languages)

    # User Query Input
    user_query = st.text_area("Enter your database query:", placeholder="Ask a question about the database...")

    if st.button("Generate SQL Query"):
        if not user_query:
            st.error("Please enter a query")
            return

        try:
            # Generate SQL Query
            sql_query = generate_sql_query(user_query)
            st.subheader("Generated SQL Query")
            st.code(sql_query, language="sql")

            # Execute SQL Query
            result = execute_sql_query(sql_query)
            st.subheader("Query Results")
            st.write(result)

            # Generate Visualization
            fig = generate_visualization(result, user_query)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")

def generate_sql_query(question):
    """Generate SQL query from natural language"""
    query_prompt = f"""
    You are an SQL expert. Convert this question into an SQL query:
    Question: {question}
    """
    response = llm.invoke(query_prompt)
    match = re.search(r'SELECT.*', response.content, re.DOTALL)
    return match.group(0).strip() if match else "No valid SQL query generated."

def execute_sql_query(query):
    """Execute SQL query and return result"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error executing query: {str(e)}"

def generate_visualization(data, title):
    """Generate a bar chart based on SQL results"""
    df = pd.DataFrame(data, columns=["Category", "Value"])
    fig = px.bar(df, x="Category", y="Value", title=title, color="Category")
    return fig
