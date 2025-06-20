# Feyenoord NL2SQL Workflow

This package provides a robust, stateful, and configurable workflow for converting natural language questions about Feyenoord into executable SQL queries. It is powered by [LangChain](https://www.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/).

## Overview

The core of this package is the `WorkflowManager`, a class that orchestrates a graph of nodes to process a query from start to finish. The workflow includes the following key steps:

1.  **Schema Caching**: Retrieves and caches the database schema.
2.  **Entity Resolution**: Identifies and clarifies ambiguous entities (like player or team names) in the user's query.
3.  **SQL Generation**: Uses a powerful LLM to generate a SQL query based on the user's question and the database schema.
4.  **Syntax Check**: Validates the generated SQL for correctness.
5.  **Query Fixing**: If the SQL is invalid, it attempts to fix it automatically.
6.  **Execution**: Runs the valid SQL against the database.
7.  **Answer Formatting (Optional)**: Formats the raw database results into a natural language response.

## Installation

This package is intended to be used as a local dependency. You can install it in your project's virtual environment in editable mode:

```bash
pip install -e /path/to/feyod-nl2sql
```

Alternatively, you can install it directly from its Git repository:

```bash
pip install git+https://github.com/jeroenvdmeer/feyod-nl2sql.git
```

## Basic Usage

Here's how to use the `WorkflowManager` in your application:

```python
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from feyod_nl2sql.workflow.manager import WorkflowManager

# Load environment variables (OPENAI_API_KEY, FEYOD_DATABASE_URL)
load_dotenv()

# Configuration for the workflow
app_config = {
    "FEYOD_DATABASE_URL": os.getenv("FEYOD_DATABASE_URL"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

async def main():
    # Initialize for natural language output
    chatbot_workflow = WorkflowManager(config=app_config, format_output=True).get_graph()

    # Initialize for raw data output
    data_workflow = WorkflowManager(config=app_config, format_output=False).get_graph()

    # --- Example Invocation ---
    question = "Hoeveel doelpunten heeft Gimenez gemaakt voor Feyenoord?"
    initial_state = {"messages": [HumanMessage(content=question)]}

    # Get a natural language answer
    final_state_chatbot = await chatbot_workflow.ainvoke(initial_state)
    answer = final_state_chatbot["messages"][-1].content
    print(f"Chatbot Answer: {answer}")

    # Get raw JSON data
    final_state_data = await data_workflow.ainvoke(initial_state)
    results = final_state_data["messages"][-1].content
    print(f"Raw Data: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

The `WorkflowManager` can be initialized with a configuration dictionary. The following keys are supported:

-   `FEYOD_DATABASE_URL`: The SQLAlchemy connection string for the database.
-   `OPENAI_API_KEY`: Your OpenAI API key.
-   `MAX_SQL_FIX_ATTEMPTS`: The maximum number of times the workflow should attempt to fix an invalid SQL query (default: 1).