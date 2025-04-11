import streamlit as st
from langchain_community.utilities import SQLDatabase
import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
from sql_components import setup_database
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import re
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import json
import datetime

# Setup database connection
db = setup_database()
if db is None:
    st.error("Failed to connect to the database")
else:
    print("Database is connected:", db)

# Define the state for our LangGraph workflow
class GraphState(TypedDict):
    question: str
    language: str
    sql_query: str
    query_result: str
    final_answer: str
    fig: go.Figure

llm = ChatGroq(model="qwen-2.5-32b", temperature=0)

# Create the graph
workflow = StateGraph(GraphState)

# Prompt for translating to English
def create_translation_to_english_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate the given text from {input_language} to English, preserving the original meaning and context."),
        ("human", "Text to translate: {input_text}")
    ])

def detect_language(state: GraphState):
    input_language = state.get('language', 'English')
    if input_language.lower() != 'english':
        translation_prompt = create_translation_to_english_prompt()
        translator = translation_prompt | llm
        translation_result = translator.invoke({
            "input_language": input_language,
            "input_text": state['question']
        })
        return {"question": translation_result.content}
    return {"question": state['question']}

# SQL query generation prompt
def create_sql_query_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL query generator. 
        Given a question in English, generate a syntactically correct SQL query. 
        Always limit results to 10 rows. Use only relevant columns. 
        
        Available Tables: {table_info}
        Question: {question}
        """)
    ])

def generate_sql_query(state: GraphState):
    sql_query_prompt = create_sql_query_prompt()
    query_generator = sql_query_prompt | llm
    result = query_generator.invoke({
        "table_info": db.get_table_info(),
        "question": state['question']
    })
    match = re.search(r'```sql\n(.*?)```', result.content, re.DOTALL)
    if match:
        return {"sql_query": match.group(1).strip()}
    match = re.search(r'SELECT.*', result.content, re.DOTALL)
    if match:
        return {"sql_query": match.group(0).strip()}

def execute_sql_query(state: GraphState):
    try:
        query_tool = QuerySQLDatabaseTool(db=db)
        result = query_tool.invoke(state['sql_query'])
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting numeric database query results into clear, professional bullet-point format. Answer directly to user's question."""),
            ("human", """Given the following:
            - Original Question: {question}
            - Query Result: {result}
            
            Provide the summary in a professional bullet-point format.""")
        ])
        response_chain = response_prompt | llm
        final_response = response_chain.invoke({
            "question": state['question'],
            "result": result
        }).content
        return {"query_result": final_response}
    except Exception as e:
        return {"query_result": f"Error executing query: {str(e)}"}

def create_translation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Translate the following SQL query result to {language} in a professional bullet-point format."),
        ("human", "Result to translate: {query_result}")
    ])

def translate_result(state: GraphState):
    translation_prompt = create_translation_prompt()
    translator = translation_prompt | llm
    result = translator.invoke({
        "language": state['language'],
        "query_result": state['query_result']
    })
    return {"final_answer": result.content}

def create_visualization(df, chart_type, title):
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least two columns")

    if chart_type == 'bar':
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title, color=df.columns[0])
    elif chart_type == 'pie':
        fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title, hole=0.3)
    elif chart_type == 'line':
        fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title, markers=True)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig.update_layout(template='plotly_white', font=dict(size=14), title_font_size=20)
    return fig
def generate_visualization_node(state: GraphState):
    try:
        print("Generating visualization...")
        result = state['query_result']
        question = state['question']

        # Prompt to LLM
        prompt_text = f"""You are an expert data visualization consultant.

        Database Query Result: {result}
        Original Question: {question}

        Provide a JSON response with:
        1. Recommended chart type (bar/pie/line)
        2. DataFrame structure for visualization

        Respond EXACTLY in this format:
        {{
            "chart_type": "bar",
            "dataframe": {{
                "columns": ["Category", "Value"],
                "data": [["Products", 10], ["Services", 5]]
            }}
        }}
        """

        response = llm.invoke(prompt_text)
        print("LLM Raw Response:", response.content)

        # Try parsing JSON
        visualization_details = json.loads(response.content)

        # Validate keys
        if 'chart_type' not in visualization_details or 'dataframe' not in visualization_details:
            raise ValueError("Invalid visualization response format")

        df = pd.DataFrame(
            visualization_details['dataframe']['data'],
            columns=visualization_details['dataframe']['columns']
        )

        # Generate chart title
        title_response = llm.invoke(f"Generate a title for the chart based on the question: {question} in {state['language']}")
        title = title_response.content.strip().replace('"', '')

        fig = create_visualization(df, visualization_details['chart_type'], title)

        return {"fig": fig}
    
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
    except Exception as e:
        print(f"Visualization generation failed: {e}")

    return {"fig": None}


# LangGraph node setup
workflow.add_node("detect_language", detect_language)
workflow.add_node("generate_sql_query", generate_sql_query)
workflow.add_node("execute_sql_query", execute_sql_query)
workflow.add_node("translate_result", translate_result)
workflow.add_node("generate_visualization", generate_visualization_node)
workflow.add_edge(START, "detect_language")
workflow.add_edge("detect_language", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_sql_query")
workflow.add_edge("execute_sql_query", "translate_result")
workflow.add_edge("translate_result", "generate_visualization")
workflow.add_edge("generate_visualization", END)

graph_builder = workflow.compile()

# 🔁 Recent Queries Helpers
def load_recent_queries():
    if "recent_queries" not in st.session_state:
        st.session_state.recent_queries = []
    return st.session_state.recent_queries

def save_recent_query(question, result_summary, timestamp=None):
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.recent_queries.insert(0, {
        "timestamp": timestamp,
        "question": question,
        "summary": result_summary
    })
    st.session_state.recent_queries = st.session_state.recent_queries[:5]  # Keep last 5

# 🌐 Streamlit App UI
def main():
    st.set_page_config(page_title="SQL Assistant", page_icon="🤖")
    st.title("SQL Assistant🖇️🦜")

    # Sidebar for configuration
    st.sidebar.header("Choose Your Language")
    languages = ["English", "Spanish", "French", "German", "Chinese", "Arabic", "Hindi", "Portuguese", "Tamil"]
    selected_language = st.sidebar.selectbox("Select Output Language", languages)

    # Query input
    user_query = st.text_area("Enter your database query:", placeholder="Ask a question about the database...")

    if st.button("Get SQL Answer"):
        if not user_query:
            st.error("Please enter a query")
            return

        try:
            initial_state = {
                "question": user_query,
                "language": selected_language,
                "sql_query": "",
                "query_result": "",
                "final_answer": ""
            }
            final_result = graph_builder.invoke(initial_state)

            if final_result:
                st.subheader("SQL Query")
                st.write(final_result.get('sql_query', 'No SQL query generated'))

                st.subheader(final_result.get('final_answer', 'No result available'))
                fig = final_result.get('fig')
                if fig and isinstance(fig, go.Figure):
                    st.plotly_chart(fig)
                else:
                    st.warning("No visualization available for this query.")


                # Save query to recent
                save_recent_query(user_query, final_result.get('final_answer', 'No result'))

            else:
                st.error("Failed to process the query")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Show recent queries
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Recent Queries")
    for q in load_recent_queries():
        st.sidebar.markdown(f"**{q['timestamp']}**\n\n- _{q['question']}_\n\n> {q['summary'][:80]}...")

# Run the app
if __name__ == "__main__":
    main()


