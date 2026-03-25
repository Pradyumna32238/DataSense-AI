from config import Config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re

import logging

logger = logging.getLogger(__name__)

class PlannerAgent:
    def __init__(self, provider='google', model_name=None, google_api_key=None, cohere_api_key=None):
        logger.info(f"Initializing PlannerAgent with provider: {provider}")
        self.llm = Config.get_llm(
            provider=provider, 
            model_name=model_name,
            google_api_key=google_api_key, 
            cohere_api_key=cohere_api_key
        )
        self.prompt_template = PromptTemplate(
            input_variables=['query', 'history', 'table_name', 'table_info', 'example_rows'],
            template='''You are an expert data analysis planner. Your job is to determine if a user's query is relevant to the provided data context, and if so, create a concise, step-by-step plan to answer it and suggest a chart type.

# Context
You have access to a table with the following details:
- **Table Name:** `{table_name}`
- **Table Info (Columns and Types):**
{table_info}
- **Example Rows:**
{example_rows}

# Conversation History
{history}

# User Query
**{query}**

# Instructions
1.  **Relevance Check:** First, determine if the user's query can be answered using the provided table context. The query must be about the data in the table.
2.  **Output Generation:**
    *   If the query is **not relevant**, provide a JSON object with "is_relevant" set to `false`, an empty "plan" list, and "chart_type" set to "table".
    *   If the query is **relevant**, provide a JSON object with "is_relevant" set to `true`, and include a "plan" and a "chart_type".
3.  **Plan Creation (if relevant):**
    *   Create a high-level, step-by-step plan. The plan should be a list of simple, natural language instructions for another AI.
    *   When the user asks for the 'most' or 'least' of something, include a step to limit the results (e.g., "show the top 10 results").
4.  **Chart Suggestion (if relevant):**
    *   Suggest the most appropriate chart type: "bar", "line", "scatter", "pie", "histogram", "heatmap", or "table".
    *   Default to "table" if no aggregation or visualization is needed.
5.  **Final Output:** Provide your output as a single, valid JSON object with three keys: "is_relevant" (boolean), "plan" (a list of strings), and "chart_type" (a string).

**JSON Output Example (Relevant Query):**
```json
{{
  "is_relevant": true,
  "plan": [
    "Count the occurrences of each [ColumnName]",
    "Order the results by the count in descending order",
    "Show the top 10 results"
  ],
  "chart_type": "bar"
}}
```

**JSON Output Example (Irrelevant Query):**
```json
{{
  "is_relevant": false,
  "plan": [],
  "chart_type": "table"
}}
```
'''
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _extract_json(self, text: str) -> dict:
        """Extracts a JSON object from a string, even if it's embedded in markdown."""
        # Regex to find a JSON object within ```json ... ``` or just as a plain object
        match = re.search(r'''```json\s*(\{.*?\})\s*```|(\{.*?\})''', text, re.DOTALL)
        if match:
            # Prioritize the first capturing group (markdown), then the second (plain object)
            json_str = match.group(1) or match.group(2)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Handle cases where the regex is too greedy and captures invalid JSON
                logger.warning(f"Could not decode JSON: {json_str}")
                return {}
        return {}

    def create_plan(self, query: str, history: list, table_name: str, table_info: str, example_rows: str) -> dict:
        """Creates a plan and suggests a chart type based on the query and data context."""
        formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        response = self.chain.invoke({
            'query': query, 
            'history': formatted_history, 
            'table_name': table_name,
            'table_info': table_info,
            'example_rows': example_rows
        })
        
        output = self._extract_json(response)

        # Default to irrelevant if JSON is empty or relevance is false
        if not output or not output.get('is_relevant'):
            if not output:
                 logger.warning(f"Could not extract valid JSON from planner response: {response}")
            return {"is_relevant": False, "plan": [], "chart_type": "table"}

        # Sanitize the output for a relevant query
        plan = output.get('plan', [])
        chart_type = output.get('chart_type', 'table')
        if chart_type not in ['bar', 'line', 'scatter', 'pie', 'histogram', 'table']:
            chart_type = 'table'

        return {
            "is_relevant": True,
            "plan": plan,
            "chart_type": chart_type
        }