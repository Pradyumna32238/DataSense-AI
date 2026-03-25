import matplotlib
matplotlib.use('Agg') # This line must be at the very top
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.prompts import PromptTemplate
import base64
from io import BytesIO
import logging
import os
import uuid
import json
from config import Config
from utils.cache import cache
import seaborn as sns

logger = logging.getLogger(__name__)

class VisualizationAgent:
    def __init__(self, provider='google', model_name=None, google_api_key=None, cohere_api_key=None):
        logger.info("Initializing VisualizationAgent.")
        self.model_name = model_name
        self.llm = None
        
        # Initialize LLM for fallback axis selection
        try:
            self.llm = Config.get_llm(
                provider=provider, 
                model_name=model_name,
                google_api_key=google_api_key, 
                cohere_api_key=cohere_api_key
            )
        except Exception as e:
            logger.warning(f"Could not initialize LLM for Visualization fallback: {e}")

    def _get_axes_from_llm(self, df: pd.DataFrame, chart_type: str):
        """Fallback method: Asks the LLM to determine the best X and Y axes."""
        if not self.llm:
            logger.warning("No LLM available for visualization fallback.")
            return None, None

        logger.info("Using LLM fallback to determine chart axes.")
        columns_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
        sample_data = df.head(3).to_dict(orient='records')

        prompt = PromptTemplate.from_template('''
        You are a data visualization expert. We need to create a {chart_type} chart.
        
        Here are the columns and their data types:
        {columns_info}
        
        Here is a sample of the data:
        {sample_data}
        
        Determine the best column to use for the X-axis and the best column for the Y-axis.
        The Y-axis MUST be a numeric column.
        
        Return ONLY a valid JSON object with the keys "x_axis" and "y_axis". Do not include markdown formatting.
        Example: {{"x_axis": "Date", "y_axis": "Sales"}}
        ''')

        try:
            formatted_prompt = prompt.format(
                chart_type=chart_type,
                columns_info=columns_info,
                sample_data=json.dumps(sample_data)
            )
            response = self.llm.invoke(formatted_prompt)
            
            # The response from the LLM might be a string or an AIMessage object
            content = response if isinstance(response, str) else response.content
            content = content.strip()

            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
                
            result = json.loads(content)
            return result.get("x_axis"), result.get("y_axis")
        except Exception as e:
            logger.error(f"LLM axis selection failed: {e}")
            return None, None

    def _create_scatter_plot(self, df, x_axis, y_axis):
        """Creates a scatter plot."""
        df.plot(kind='scatter', x=x_axis, y=y_axis, ax=plt.gca())

    def _create_pie_chart(self, df, x_axis, y_axis):
        """Helper to prepare data for a pie chart."""
        if df[x_axis].nunique() > 10: # Limit slices for readability
            plot_df = df.groupby(x_axis)[y_axis].sum().nlargest(10)
        else:
            plot_df = df.set_index(x_axis)[y_axis]
        plot_df.plot(kind='pie', autopct='%1.1f%%', ax=plt.gca(), legend=False)

    def _create_histogram(self, df, x_axis, y_axis=None): # y_axis is not used but kept for consistency
        """Creates a histogram."""
        df[x_axis].plot(kind='hist', ax=plt.gca(), bins=20)

    def _create_heatmap(self, df, x_axis, y_axis, value_col):
        """Creates a heatmap from the dataframe."""
        pivot_table = df.pivot(index=y_axis, columns=x_axis, values=value_col)
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", ax=plt.gca())

    def generate_visualization(self, execution_result: dict, chart_type: str = 'bar', dataset_hash: str = '', sql_query: str = '') -> dict:
        logger.info(f"Generating visualization with chart type: {chart_type}")

        # Check cache first
        cache_key = cache.get_chart_key(dataset_hash, sql_query, chart_type)
        cached_chart = cache.get(cache_key)
        if cached_chart:
            logger.info(f"Cache hit for chart: {chart_type}")
            return cached_chart
        
        if 'result' not in execution_result or not execution_result['result']:
            return {"error": "No result to visualize."}
        
        try:
            df = pd.DataFrame(execution_result['result'])
            if df.empty:
                return {"error": "Result is empty."}

            # Handle table generation
            if chart_type == 'table':
                logger.info("Handling table generation. Returning structured data.")
                table_data = df.to_dict(orient='split')
                result = {"table": table_data}
                cache.set(cache_key, result)
                return result

            # --- Axis Selection Logic ---
            x_axis, y_axis, value_col = None, None, None

            # 1. Heuristic-based selection
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

            if chart_type in ['bar', 'line', 'scatter', 'pie']:
                if numeric_cols and categorical_cols:
                    x_axis = categorical_cols[0]
                    y_axis = numeric_cols[0]
                elif len(numeric_cols) >= 2 and chart_type == 'scatter':
                    x_axis = numeric_cols[0]
                    y_axis = numeric_cols[1]
            elif chart_type == 'histogram':
                if numeric_cols:
                    x_axis = numeric_cols[0]
            elif chart_type == 'heatmap':
                if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                    x_axis = categorical_cols[0]
                    y_axis = categorical_cols[1]
                    value_col = numeric_cols[0]

            # 2. LLM Fallback if heuristics fail
            if (chart_type not in ['histogram', 'heatmap'] and (not x_axis or not y_axis)) or \
               (chart_type == 'histogram' and not x_axis) or \
               (chart_type == 'heatmap' and (not x_axis or not y_axis or not value_col)):
                logger.warning("Heuristic axis selection failed. Attempting LLM fallback.")
                # Note: Current LLM fallback only supports x and y axis. Heatmap will rely on heuristics.
                if chart_type != 'heatmap':
                    x_axis, y_axis = self._get_axes_from_llm(df, chart_type)
                
                if not x_axis or (chart_type not in ['histogram', 'heatmap'] and not y_axis) or x_axis not in df.columns or (y_axis and y_axis not in df.columns):
                    logger.error("Axis selection failed.")
                    return {"error": "Could not determine appropriate axes for the chart."}

            # Handle cases with multiple categorical columns by combining them
            if len(categorical_cols) > 1 and x_axis in categorical_cols and chart_type not in ['pie', 'histogram', 'heatmap']:
                x_axis_label = ' & '.join(categorical_cols)
                df[x_axis_label] = df[categorical_cols].apply(lambda x: ' - '.join(x.astype(str)), axis=1)
                x_axis = x_axis_label

            plt.figure(figsize=(12, 7))

            chart_functions = {
                'bar': lambda: df.plot(kind='bar', x=x_axis, y=y_axis, ax=plt.gca()),
                'line': lambda: df.plot(kind='line', x=x_axis, y=y_axis, ax=plt.gca()),
                'scatter': lambda: self._create_scatter_plot(df, x_axis, y_axis),
                'pie': lambda: self._create_pie_chart(df, x_axis, y_axis),
                'histogram': lambda: self._create_histogram(df, x_axis),
                'heatmap': lambda: self._create_heatmap(df, x_axis, y_axis, value_col)
            }

            # Get the function from the dictionary and call it
            draw_chart = chart_functions.get(chart_type)
            if draw_chart:
                draw_chart()
            else:
                # Default to bar chart if type is unknown
                df.plot(kind='bar', x=x_axis, y=y_axis, ax=plt.gca())

            plt.title('Analysis Result')
            plt.xlabel(x_axis)
            if chart_type == 'pie':
                plt.ylabel('')
            elif chart_type == 'histogram':
                plt.ylabel('Frequency')
            else:
                plt.ylabel(y_axis)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # --- Save image to file ---
            image_dir = os.path.join('static', 'images')
            os.makedirs(image_dir, exist_ok=True)
            
            image_filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(image_dir, image_filename)
            
            plt.savefig(image_path, format="png")
            plt.close()

            # Return a URL-friendly path
            url_path = f"/static/images/{image_filename}"
            result = {"visualization": url_path}
            cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error("Failed to generate plot", exc_info=True)
            return {"error": f"Failed to generate plot: {e}"}