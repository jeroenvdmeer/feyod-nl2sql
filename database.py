"""Database connection and interaction logic using SQLAlchemy for a sync SQLite connection."""

import logging
import asyncio
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from common import config
from common.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Global engine and toolkit instances
_sql_database = None
_toolkit = None
_executor = ThreadPoolExecutor()

def get_sql_database():
    """Initializes and returns the SQLDatabase instance (singleton)."""
    global _sql_database
    if _sql_database is None:
        # Use a synchronous engine for LangChain SQLDatabase
        # Convert async URL to sync if needed
        db_url = config.FEYOD_DATABASE_URL.replace("+aiosqlite", "")
        try:
            logger.info(f"Creating SQLDatabase instance for URL: {db_url}")
            sync_engine = create_engine(db_url)
            _sql_database = SQLDatabase(sync_engine)
        except Exception as e:
            logger.exception(f"Failed to create SQLDatabase instance: {e}")
            raise
    return _sql_database

def get_toolkit():
    """Initializes and returns the SQLDatabaseToolkit instance (singleton)."""
    global _toolkit
    if (_toolkit is None):
        db = get_sql_database()
        llm = get_llm()
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        _toolkit = toolkit.get_tools()
    return _toolkit

async def get_schema_description() -> str:
    """Retrieves a string description of the database schema using toolkit tools."""
    loop = asyncio.get_running_loop()
    toolkit = get_toolkit()
    try:
        def run_schema():
            tables_tool = next(tool for tool in toolkit if tool.name == "sql_db_list_tables")
            schema_tool = next(tool for tool in toolkit if tool.name == "sql_db_schema")
            tables = tables_tool.run("")
            return schema_tool.run(tables)
        schema = await loop.run_in_executor(_executor, run_schema)
        logger.info("Successfully retrieved schema description.")
        logger.debug(f"Schema: {schema}")
        return schema
    except Exception as e:
        logger.error(f"Error retrieving schema description: {e}")
        return f"Error retrieving schema description: {e}"

async def validate_query(sql: str) -> str:
    """Validates a given SQL query using QuerySQLCheckerTool from the toolkit."""
    loop = asyncio.get_running_loop()
    toolkit = get_toolkit()
    try:
        def run_validate():
            checker_tool = next(tool for tool in toolkit if tool.name == "sql_db_query_checker")
            return checker_tool.run(sql)
        validation_result = await loop.run_in_executor(_executor, run_validate)
        if "Error" in validation_result:
            raise ValueError(f"Query validation failed: {validation_result}")
        logger.info("Query validation successful.")
        return validation_result
    except Exception as e:
        logger.error(f"Error validating query: {e}")
        raise

async def execute_query(sql: str) -> str:
    """Executes a given SQL query using the toolkit's query tool and returns results as JSON."""
    loop = asyncio.get_running_loop()
    toolkit = get_toolkit()
    try:
        def run_query():
            query_tool = next(tool for tool in toolkit if tool.name == "sql_db_query")
            return query_tool.run(sql)
        results = await loop.run_in_executor(_executor, run_query)
        logger.info(f"Query executed successfully: {results}")
        return results
    except Exception as e:
        logger.error(f"Unexpected error executing query: {e}")
        raise ValueError(f"Unexpected error executing query: {e}")

async def get_distinct_player_names() -> list[str]:
    """Return a list of all distinct player names from the players table."""
    loop = asyncio.get_running_loop()
    db_url = config.FEYOD_DATABASE_URL.replace("+aiosqlite", "")
    def run_query():
        sync_engine = create_engine(db_url)
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT playerName FROM players WHERE playerName IS NOT NULL AND playerName != ''"))
            return [row[0] for row in result.fetchall()]
    try:
        names = await loop.run_in_executor(_executor, run_query)
        return names
    except Exception as e:
        logger.error(f"Error fetching player names: {e}")
        return []

async def get_distinct_club_names() -> list[str]:
    """Return a list of all distinct club names from the clubs table."""
    loop = asyncio.get_running_loop()
    db_url = config.FEYOD_DATABASE_URL.replace("+aiosqlite", "")
    def run_query():
        sync_engine = create_engine(db_url)
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT clubName FROM clubs WHERE clubName IS NOT NULL AND clubName != ''"))
            return [row[0] for row in result.fetchall()]
    try:
        names = await loop.run_in_executor(_executor, run_query)
        return names
    except Exception as e:
        logger.error(f"Error fetching club names: {e}")
        return []

# Example usage (for testing purposes) - Requires running in an async context
async def _test_main():
    logging.basicConfig(level=logging.INFO)
    # Set FEYOD_DATABASE_URL environment variable before running for this test
    if not config.FEYOD_DATABASE_URL:
        print("Please set the FEYOD_DATABASE_URL environment variable to run the test.")
        return

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