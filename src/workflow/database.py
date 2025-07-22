"""Database connection and interaction logic using SQLAlchemy."""

import logging
import asyncio
from typing import List
from sqlalchemy import create_engine, text, Engine
from concurrent.futures import ThreadPoolExecutor
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# A thread pool executor for running sync database calls in an async environment
_executor = ThreadPoolExecutor()

def get_engine(db_url: str) -> Engine:
    """Creates and returns a SQLAlchemy engine for the given database URL."""
    if not db_url:
        raise ValueError("Database URL is required to create an engine.")
    # LangChain's SQLDatabase uses a sync engine.
    sync_db_url = db_url.replace("+aiosqlite", "")
    logger.info(f"Creating SQLAlchemy engine for URL: {sync_db_url}")
    return create_engine(sync_db_url)

def get_sql_database(engine: Engine) -> SQLDatabase:
    """Initializes and returns the SQLDatabase instance."""
    try:
        logger.info("Creating SQLDatabase instance from engine.")
        return SQLDatabase(engine)
    except Exception as e:
        logger.exception(f"Failed to create SQLDatabase instance: {e}")
        raise

def get_toolkit(db: SQLDatabase, llm: BaseChatModel) -> List:
    """Initializes and returns the SQLDatabaseToolkit instance."""
    try:
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        return toolkit.get_tools()
    except Exception as e:
        logger.exception("Failed to create SQLDatabaseToolkit.")
        raise

async def get_schema_description(db_url: str, llm: BaseChatModel) -> str:
    """Retrieves a string description of the database schema."""
    loop = asyncio.get_running_loop()
    try:
        engine = get_engine(db_url)
        db = get_sql_database(engine)
        toolkit = get_toolkit(db, llm)
        
        def run_schema_tools():
            tables_tool = next(tool for tool in toolkit if tool.name == "sql_db_list_tables")
            schema_tool = next(tool for tool in toolkit if tool.name == "sql_db_schema")
            table_names = tables_tool.run("")
            return schema_tool.run(table_names)
            
        schema = await loop.run_in_executor(_executor, run_schema_tools)
        logger.info("Successfully retrieved schema description.")
        return schema
    except Exception as e:
        logger.exception("Error retrieving schema description.")
        return f"Error: Could not retrieve schema. Details: {e}"

async def execute_query(sql: str, db_url: str) -> List[dict]:
    """Executes a given SQL query and returns results as a list of dictionaries."""
    loop = asyncio.get_running_loop()
    try:
        engine = get_engine(db_url)
        
        def run_sync_query():
            with engine.connect() as connection:
                result = connection.execute(text(sql))
                return [row._asdict() for row in result]

        results = await loop.run_in_executor(_executor, run_sync_query)
        logger.info(f"Query executed successfully. Returned {len(results)} rows.")
        return results
    except Exception as e:
        logger.exception(f"Unexpected error executing query: {sql}")
        raise ValueError(f"Query execution failed: {e}")

async def get_distinct_values(column_name: str, table_name: str, db_url: str) -> list[str]:
    """Return a list of all distinct values from a column in a table."""
    loop = asyncio.get_running_loop()
    
    def run_query():
        engine = get_engine(db_url)
        with engine.connect() as conn:
            # Use bound parameters to prevent SQL injection, even with internal values
            query = text(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL AND {column_name} != ''")
            result = conn.execute(query)
            return [row[0] for row in result.fetchall()]
            
    try:
        names = await loop.run_in_executor(_executor, run_query)
        return names
    except Exception as e:
        logger.error(f"Error fetching distinct values for {column_name} from {table_name}: {e}")
        return []

async def check_sql_syntax(sql_query: str, db_url: str) -> tuple[bool, str | None]:
    """
    Checks the syntax of an SQLite query using EXPLAIN via SQLAlchemy engine.
    Returns (True, None) on success, (False, error_message) on failure.
    """
    loop = asyncio.get_running_loop()
    try:
        engine = get_engine(db_url)
        def run_explain():
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"EXPLAIN {sql_query}"))
                return True, None
            except Exception as e:
                logger.warning(f"SQL syntax check failed for query '{sql_query}': {e}")
                return False, str(e)
        is_valid, error = await loop.run_in_executor(_executor, run_explain)
        if is_valid:
            logger.info(f"SQL syntax check passed for: {sql_query}")
        return is_valid, error
    except Exception as e:
        logger.exception(f"Unexpected error during SQL syntax check: {e}")
        return False, f"Unexpected error during syntax check: {e}"

# Example usage (for testing purposes) - Requires running in an async context
async def _test_main():
    logging.basicConfig(level=logging.INFO)

    print("Testing SQLDatabase connection...")
    try:
        get_sql_database()
        print("SQLDatabase instance created successfully.")
    except Exception as e:
        print(f"Connection failed: {e}")

    print("\nFetching schema...")
    try:
        schema = await get_schema_description()
        print(schema)
    except Exception as e:
        print(f"Schema fetch failed: {e}")

    print("\nTesting query execution (example: first 5 clubs)...")
    try:
        clubs = await execute_query("SELECT clubId, clubName FROM clubs LIMIT 5;")
        print(clubs)
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    import asyncio
    print("Running database test function. Ensure FEYOD_DATABASE_URL is set in your environment.")
    asyncio.run(_test_main())