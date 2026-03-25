import pandas as pd

from langgraph.graph import StateGraph, END
from .state import AgentState
from agents.planner_agent import PlannerAgent
from agents.sql_agent import SQLAgent
from execution.sql_executor import execute_sql
from agents.visualization_agent import VisualizationAgent
from agents.summary_agent import SummaryAgent
from utils.cache import semantic_cache

def semantic_cache_node(state: AgentState) -> AgentState:
    """Checks the semantic cache for a similar query."""
    cached_response = semantic_cache.search(state['query'])
    if cached_response:
        return {**state, **cached_response, "semantic_cache_hit": True}
    return {**state, "semantic_cache_hit": False}

def schema_analysis_node(state: AgentState) -> AgentState:
    """Analyzes the data to extract schema, column descriptions, and example rows."""
    try:
        print("---ANALYZING SCHEMA---")
        data_path = state['data_path']
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            return {**state, "error": f"Unsupported file type: {data_path}"}

        # Get table info (column names and types)
        table_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
        
        # Get example rows
        example_rows = df.head(3).to_string()

        return {
            **state, 
            "table_info": table_info, 
            "example_rows": example_rows, 
            "error": None
        }
    except Exception as e:
        return {**state, "error": f"Error in Schema Analysis: {e}"}

def planner_node(state: AgentState) -> AgentState:
    try:
        print("---PLANNING---")
        planner_agent = PlannerAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        plan_and_chart_type = planner_agent.create_plan(state['query'], state['history'], state['table_name'], state['table_info'], state['example_rows'])
        return {**state, **plan_and_chart_type, "error": None}
    except Exception as e:
        return {**state, "error": f"Error in Planner: {e}"}

def sql_generator_node(state: AgentState) -> AgentState:
    if state.get('error'): return state
    try:
        print("---GENERATING SQL---")
        sql_agent = SQLAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        sql_query = sql_agent.generate_sql(state['table_name'], state['table_info'], state['example_rows'], state['plan'])
        return {**state, "sql_query": sql_query, "error": None}
    except Exception as e:
        return {**state, "error": f"Error in SQL Generator: {e}"}
def code_executor_node(state: AgentState) -> AgentState:
    if state.get('error'): return state
    try:
        print("---EXECUTING CODE---")
        execution_result = execute_sql(state['sql_query'], state['data_path'], state['table_name'], state['dataset_hash'])
        if "error" in execution_result:
            return {**state, "error": f"Error in Code Execution: {execution_result['error']}"}
        return {**state, "execution_result": execution_result, "error": None}
    except Exception as e:
        return {**state, "error": f"Error in Code Executor: {e}"}

def visualization_node(state: AgentState) -> AgentState:
    if state.get('error'): return state
    try:
        print("---GENERATING VISUALIZATION---")
        visualization_agent = VisualizationAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        visualization_output = visualization_agent.generate_visualization(state['execution_result'], state['chart_type'])
        
        if "error" in visualization_output:
            return {**state, "error": f"Error in Visualization: {visualization_output['error']}"}
        
        # Update state with either visualization path or table data
        updated_state = {**state, "error": None}
        if 'visualization' in visualization_output:
            updated_state['visualization'] = visualization_output.get('visualization')
        if 'table' in visualization_output:
            updated_state['table'] = visualization_output.get('table')
            
        return updated_state

    except Exception as e:
        return {**state, "error": f"Error in Visualizer: {e}"}

def summary_node(state: AgentState) -> AgentState:
    if state.get('error'): return state
    try:
        print("---GENERATING SUMMARY---")
        summary_agent = SummaryAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        summary_text = summary_agent.generate_summary(state['query'], state['execution_result'], state.get('dataset_hash'), state['sql_query'])
        return {**state, "summary": summary_text, "error": None}
    except Exception as e:
        return {**state, "error": f"Error in Summarizer: {e}"}

def should_generate_visualization(state: AgentState) -> str:
    """Determines whether to generate a visualization or a summary."""
    if state.get('error'):
        return "end"
    
    print("---ROUTING---")
    chart_type = state.get('chart_type', 'none').lower()
    print(f"Chart type for routing: '{chart_type}'")
    
    if chart_type != 'none' and chart_type is not None:
        print("Decision: Route to VISUALIZER")
        return "visualizer"
    else:
        print("Decision: Route to SUMMARIZER")
        return "summarizer"



def rejection_node(state: AgentState) -> AgentState:
    """Ends the process if the query is irrelevant."""
    print("---QUERY REJECTED---")
    return {**state, "summary": "I can only answer questions related to the uploaded data. Please ask a question about the available columns and data."}


def route_after_planner(state: AgentState) -> str:
    """Determines whether to continue with the analysis or stop based on the planner's output."""
    print("---ROUTING BASED ON RELEVANCE---")
    if state.get("error"):
        return "end"
    
    is_relevant = state.get("is_relevant", False)
    if is_relevant:
        print("Decision: Query is relevant. Route to SQL_GENERATOR.")
        return "sql_generator"
    else:
        print("Decision: Query is irrelevant. Route to REJECTION.")
        return "rejection"

def route_after_semantic_cache(state: AgentState) -> str:
    """Routes the workflow based on the semantic cache result."""
    if state.get("semantic_cache_hit"):
        return "end"
    return "schema_analyzer"

def get_graph_app():
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("semantic_cache", semantic_cache_node)
    workflow.add_node("schema_analyzer", schema_analysis_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("visualizer", visualization_node)
    workflow.add_node("summarizer", summary_node)
    workflow.add_node("rejection", rejection_node)

    # Add edges
    workflow.set_entry_point("semantic_cache")

    workflow.add_conditional_edges(
        "semantic_cache",
        route_after_semantic_cache,
        {
            "end": END,
            "schema_analyzer": "schema_analyzer"
        }
    )

    workflow.add_edge("schema_analyzer", "planner")

    # Add conditional routing based on relevance
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "sql_generator": "sql_generator",
            "rejection": "rejection",
            "end": END
        }
    )
    
    workflow.add_edge("sql_generator", "code_executor")

    # Add conditional routing for visualization
    workflow.add_conditional_edges(
        "code_executor",
        should_generate_visualization,
        {
            "visualizer": "visualizer",
            "summarizer": "summarizer",
            "end": END
        }
    )

    workflow.add_edge("visualizer", "summarizer") # Always generate a summary after visualization
    workflow.add_edge("summarizer", END)

    # Compile the graph
    return workflow.compile()

graph_app = get_graph_app()