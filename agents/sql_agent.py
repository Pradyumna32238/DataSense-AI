from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from config import Config
from langchain_core.output_parsers import StrOutputParser
import re
import logging

logger = logging.getLogger(__name__)

class SQLAgent:
    def __init__(self, provider: str = 'google', model_name: str = None, google_api_key: str = None, cohere_api_key: str = None):
        logger.info(f"Initializing SQLAgent with provider: {provider}")
        self.llm = Config.get_llm(
            provider=provider, 
            model_name=model_name,
            google_api_key=google_api_key, 
            cohere_api_key=cohere_api_key
        )
        self.prompt_template = PromptTemplate(
            input_variables=['table_name', 'table_info', 'example_rows', 'plan'],
            template=(
                "You are an expert DuckDB data analyst. Your only purpose is to write a single, syntactically correct DuckDB query that accomplishes the goal of the provided plan."
                "\n# Instructions:"
                "\n1. **Your ONLY task is to translate the following high-level plan into a single, valid DuckDB query.**"
                "\n2. **Use the provided table context to inform the query.**"
                "\n3. **The query MUST be a single statement.**"
                "\n4. **The query must be directly executable on a DuckDB database.**"
                
                "\n\n# DuckDB Best Practices:"
                "\n- **Date Filtering:** When filtering by year, do NOT use `STRPTIME`. This function returns a `TIMESTAMP`, which will cause a type error when compared to an integer year. Instead, cast the date column to `DATE` and use the `YEAR()` function."
                "\n  - **INCORRECT:** `... WHERE STRPTIME(\"Date\", '%Y') BETWEEN 1995 AND 2005`"
                "\n  - **CORRECT:** `... WHERE YEAR(CAST(\"Date\" AS DATE)) BETWEEN 1995 AND 2005`"

                "\n\n# Context:"
                "\nYou will be querying a table with the following details:"
                "\n- **Table Name:** `{table_name}`"
                "\n- **Table Info (Columns and Types):**\n{table_info}"
                "\n- **Example Rows:**\n{example_rows}"
                "\n- **Execution Plan:**\n{plan}\n"
                "\n# DuckDB Query:"
            )
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _extract_sql(self, text: str) -> str:
        """Extracts the SQL query from a markdown code block, supporting both 'sql' and 'duckdb'."""
        # The pattern now looks for ```sql, ```duckdb, or just ``` followed by the query.
        match = re.search(r"```(sql|duckdb)?\n(.*?)\n```", text, re.DOTALL)
        if match:
            # The actual query is in the second capturing group.
            return match.group(2).strip()
        # Fallback for cases where the model doesn't use markdown
        return text.strip()

    def generate_sql(self, table_name: str, table_info: str, example_rows: str, plan: list) -> str:
        """Generates a SQL query to answer a given query using LangChain and Ollama."""
        response = self.chain.invoke({
            'table_name': table_name, 
            'table_info': table_info, 
            'example_rows': example_rows,
            'plan': "\n".join(f"- {step}" for step in plan)
        })
        sql_query = self._extract_sql(response)
        logger.info(f"LLM-generated SQL: {sql_query}")
        return sql_query