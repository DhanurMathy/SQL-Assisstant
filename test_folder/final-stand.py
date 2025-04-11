import streamlit as st
from langchain_community.utilities import SQLDatabase
import os
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv()
from sql_components import setup_database
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import re
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import json

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
    

llm = ChatGroq(model="qwen-2.5-32b", temperature=0) #gemma2-9b-it,
# Create the graph
workflow = StateGraph(GraphState)
print("Workflow is created:", workflow)

# Translation prompt
def create_translation_to_english_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "You are a professional translator. Translate the given text from {input_language} to English, preserving the original meaning and context."),
        ("human", "Text to translate: {input_text}")
    ])
# Modified detect_language function
def detect_language(state: GraphState):
    input_language = state.get('language', 'English')
    
    if input_language.lower() != 'english':
        translation_prompt = create_translation_to_english_prompt()
        translator = translation_prompt | llm
        
        translation_result = translator.invoke({
            "input_language": input_language,
            "input_text": state['question']
        })
        
        return {
            
            "question": translation_result.content
            
        }
    
    return {"question": state['question']}

# SQL query generation prompt
def create_sql_query_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an expert SQL query generator. 
        Given a question in English, generate a syntactically correct SQL query. 
        Follow these rules:
        - Always limit results to 10 rows
        - Use only columns that are necessary to answer the question
        - Pay attention to table relationships
        - Handle potential ambiguities carefully
        
        Available Tables: {table_info}
        
        Question: {question}
        """)
    ])
def generate_sql_query(state: GraphState):
    sql_query_prompt = create_sql_query_prompt()
    query_generator = sql_query_prompt | llm
    #print("db.get_table_info():", db.get_table_info())
    result = query_generator.invoke({
        "table_info": db.get_table_info(),
        "question": state['question']
    })
    match = re.search(r'```sql\n(.*?)```', result.content, re.DOTALL)
    if match:
        # If found within code block
        return {"sql_query": match.group(1).strip() }
    match = re.search(r'SELECT.*', result.content, re.DOTALL)
    
    if match:
        return {"sql_query": match.group(0).strip() } 
    
    
def execute_sql_query(state: GraphState):
    try:
        query_tool = QuerySQLDatabaseTool(db=db)
        print("SQL Query:", state['sql_query'])
        print("Question:", state['question'])
        # Execute the SQL query
        result = query_tool.invoke(state['sql_query'])
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at converting numeric database query results 
            into clear, professional bullet-point format.Answer directly to user's question."""),
            ("human", """Given the following:
            - Original Question: {question}
            - Query Result: {result}
            
            Ensure the response is grammatically correct and easy to understand.
            Provide the summary in a professional bullet-point format.""")
        ])
        # Create a chain to generate the natural language response
        response_chain = response_prompt | llm
        # Generate the natural language response
        final_response = response_chain.invoke({
            "question": state['question'],
            "result": result
        }).content
        # Print the final response for debugging
        print("Final Response:", final_response)       
        # Convert result to string for display
        return {"query_result": final_response}
    except Exception as e:
        print("Error executing SQL query:", e)
        # Handle any exceptions that occur during query execution
        return {"query_result": f"Error executing query: {str(e)}"}
    
def create_translation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Translate the following SQL query result to {language} in a professional bullet-point format."),
        ("human", "Result to translate: {query_result}")
    ])
def translate_result(state: GraphState ):
    translation_prompt = create_translation_prompt()
    translator = translation_prompt | llm
    
    result = translator.invoke({
        "language": state['language'],
        "query_result": state['query_result']
    })
    
    return {"final_answer": result.content}
def create_visualization(df, chart_type, title):
    print(f"Title: {title}")
    # """
    # Create a Plotly visualization based on the DataFrame and chart type
    
    # Args:
    #     df (pd.DataFrame): Input DataFrame
    #     chart_type (str): Type of chart to create
    #     question (str): Original question for title context
    
    # Returns:
    #     plotly.graph_objs._figure.Figure: Plotly figure object
    # """
    # Ensure we have at least two columns
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least two columns")
    
    # Dynamic chart generation
    if chart_type == 'bar':
        fig = px.bar(
            df, 
            x=df.columns[0], 
            y=df.columns[1], 
            title=title,
            labels={
                df.columns[0]: df.columns[0],
                df.columns[1]: df.columns[1]
            },
            color=df.columns[0]
        )
    elif chart_type == 'pie':
        fig = px.pie(
            df, 
            names=df.columns[0], 
            values=df.columns[1],
            title=title,
            hole=0.3
        )
    elif chart_type == 'line':
        fig = px.line(
            df, 
            x=df.columns[0], 
            y=df.columns[1],
            title=title,
            markers=True
        )
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    
    # Improve chart aesthetics
    fig.update_layout(
        template='plotly_white',
        font=dict(size=14),
        title_font_size=20
    )
    
    return fig

def generate_visualization_node(state: GraphState):
    try:
        print("Generating visualization...")
        print("State before visualization:", state['query_result'])
        print("State before visualization:", state['question'])
        result = state['query_result']
        question = state['question']
        # Prompt to generate natural language response and chart details
        prompt_text = f"""You are an expert data visualization consultant.

        Database Query Result: {result}
        Original Question: {question}

        Provide a JSON response with:
        1. Recommended chart type (bar/pie/line)
        2. DataFrame structure for visualization

        Respond EXACTLY in this format:
        Example 1:
        {{
            "chart_type": "bar",
            "dataframe": {{
                "columns": ["Category", "Value"],
                "data": [["Products", 10]]
            }} }}
        Example 2:
           {{ 
        "chart_type": "pie",
        "dataframe": {{
            "columns": ["Category", "Value"],
            "data": [["Electronics", 40], ["Clothing", 30], ["Groceries", 20], ["Books", 10]]
        }} }}
        Example 3:
           {{
        "chart_type": "line",
        "dataframe": {{
            "columns": ["Month", "Sales"],
            "data": [["January", 100], ["February", 150], ["March", 200], ["April", 250]]
        }}}}
        """
        response = llm.invoke(prompt_text)
        visualization_details = json.loads(response.content)
        print("Visualization details:", visualization_details)
                
        # Create DataFrame
        df = pd.DataFrame(visualization_details['dataframe']['data'],columns=visualization_details['dataframe']['columns'])
        # Create visualization
        title=llm.invoke(f"Generate a title for the chart based on the question: {state['question']} in {state['language']}")
        
        pre_final_title = title.content
        final_title = pre_final_title.replace('"', '')
        print("Final title:", final_title)
        fig = create_visualization(df, visualization_details['chart_type'], final_title)
        
        # Add the figure to the state
        state['fig'] = fig
        
        return {"fig": fig}
    
    except Exception as e:
        print(f"Error: {e}")



# Add nodes
workflow.add_node("detect_language",detect_language)
workflow.add_node("generate_sql_query",generate_sql_query)
workflow.add_node("execute_sql_query", execute_sql_query)
workflow.add_node("translate_result", translate_result)
workflow.add_node("generate_visualization", generate_visualization_node)
# Define edges
workflow.add_edge(START, "detect_language")
workflow.add_edge("detect_language", "generate_sql_query")
workflow.add_edge("generate_sql_query", "execute_sql_query")
workflow.add_edge("execute_sql_query", "translate_result")
workflow.add_edge("translate_result", "generate_visualization")
workflow.add_edge("generate_visualization", END)
print("Graph is created")

graph_builder=workflow.compile()


def main():
    # Set page configuration
    st.set_page_config(page_title="SQL Assistant", page_icon="🤖")
    st.title("SQL Assistant🖇️🦜")

    # Sidebar for configuration
    st.sidebar.header("Choose Your Language")
    
    # Language selection
    languages = [
        "English", "Spanish", "French", "German", 
        "Chinese", "Arabic", "Hindi", "Portuguese","Tamil"
    ]
    selected_language = st.sidebar.selectbox(
        "Select Output Language", 
        languages
    )

    # Query input
    user_query = st.text_area("Enter your database query:", 
                               placeholder="Ask a question about the database...")
    # Submit button
    if st.button("Get SQL Answer"):
        # Validate input
        if not user_query:
            st.error("Please enter a query")
            return

        try:
            print("Input is received")
            # Prepare initial state
            initial_state = {
                "question": user_query,
                "language": selected_language,                
                "sql_query":"",
                "query_result":"",
                "final_answer":""
            }
            print("Initial state:", initial_state)
            
           
            final_result= graph_builder.invoke(initial_state)
           

            # Display results
            if final_result:
                st.subheader("SQL Query")
                st.write(final_result.get('sql_query', 'No SQL query generated'))
                
                st.subheader(final_result.get('final_answer', 'No result available'))
               # st.write(final_result.get('final_answer', 'No result available'))
                st.plotly_chart(final_result.get('fig','No figure available'))
                

            else:
                st.error("Failed to process the query")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    

# SQL Assistant Nodes (sql_assistant_nodes.py)
# Add this line to run the Streamlit app
if __name__ == "__main__":
    main()
    