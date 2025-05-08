"""Database connection and interaction logic using SQLAlchemy for async SQLite."""

import logging
import json
from typing import Dict, Optional
import sqlalchemy.exc
from sqlalchemy.ext.asyncio import create_async_engine, AsyncConnection
from sqlalchemy import text

# Import config from the common module
from . import config

logger = logging.getLogger(__name__)

# Global engine instance (lazy initialization can be added if needed)
_engine = None

def get_engine():
    """Initializes and returns the SQLAlchemy async engine."""
    global _engine
    if _engine is None:
        if not config.FEYOD_DATABASE_URL:
            logger.error("FEYOD_DATABASE_URL is not configured. Cannot create database engine.")
            raise ValueError("Database URL is not configured.")
        try:
            logger.info(f"Creating async SQLAlchemy engine for URL: {config.FEYOD_DATABASE_URL}")
            _engine = create_async_engine(config.FEYOD_DATABASE_URL)
        except Exception as e:
            logger.exception(f"Failed to create SQLAlchemy engine: {e}")
            raise
    return _engine

async def get_db_connection() -> AsyncConnection:
    """Establishes an asynchronous connection to the database via the engine."""
    engine = get_engine()
    try:
        conn = await engine.connect()
        logger.info("Successfully obtained database connection from engine.")
        return conn
    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(f"Error connecting to database: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting database connection: {e}")
        raise ValueError(f"Failed to connect to database: {e}")


async def close_db_connection(conn: Optional[AsyncConnection]):
    """Closes the database connection."""
    if conn:
        try:
            await conn.close()
            logger.info("Database connection closed.")
        except sqlalchemy.exc.SQLAlchemyError as e:
            logger.error(f"Error closing database connection: {e}")
        except Exception as e:
            logger.error(f"Unexpected error closing database connection: {e}")


async def get_schema_description() -> str:
    """Retrieves a string description of the database schema using SQLAlchemy introspection."""
    conn: Optional[AsyncConnection] = None
    try:
        conn = await get_db_connection()

        # Get table names
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        tables = result.fetchall()

        schema_parts = []
        for table in tables:
            table_name = table[0] # Access by index for Row object
            if table_name.startswith("sqlite_"): # Skip internal SQLite tables
                continue
            schema_parts.append(f"Table '{table_name}':")

            # Get column details for each table
            col_result = await conn.execute(text(f"PRAGMA table_info('{table_name}');"))
            columns = col_result.fetchall()
            for column in columns:
                # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
                col_name = column[1]
                col_type = column[2]
                col_pk = " (Primary Key)" if column[5] else ""
                schema_parts.append(f"  - {col_name}: {col_type}{col_pk}")
            schema_parts.append("")

        return "\n".join(schema_parts)

    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error retrieving schema: {e}")
        return f"Error retrieving schema: {e}"
    except Exception as e:
        logger.error(f"Unexpected error retrieving schema: {e}")
        return f"Unexpected error retrieving schema: {e}"
    finally:
        await close_db_connection(conn)


async def execute_query(sql: str, params: Optional[Dict | tuple] = None) -> str:
    """Executes a given SQL query safely with parameters and returns results as dicts."""
    conn: Optional[AsyncConnection] = None
    results: str = None

    if params is None:
        params = {} # Use empty dict for execute if no params provided

    try:
        conn = await get_db_connection()
        logger.info(f"Executing SQL: {sql} with params: {params}")
        query = text(sql)
        result_proxy = await conn.execute(query, params)

        # Check if the query returns rows
        if result_proxy.returns_rows:
            # Fetch all rows and convert them to dictionaries
            rows = result_proxy.mappings().fetchall()
            results = json.dumps([dict(row) for row in rows])
            logger.info(f"Query executed successfully: '{results}'")

    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error executing query: {e}")
        # Propagate a clear error message
        raise ValueError(f"Error executing query: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error executing query: {e}")
        raise ValueError(f"Unexpected error executing query: {e}")
    finally:
        await close_db_connection(conn)
    return results

# Example usage (for testing purposes) - Requires running in an async context
async def _test_main():
    logging.basicConfig(level=logging.INFO)
    # Set FEYOD_DATABASE_URL environment variable before running for this test
    if not config.FEYOD_DATABASE_URL:
        print("Please set the FEYOD_DATABASE_URL environment variable to run the test.")
        return

    print("Testing database connection...")
    conn = None
    try:
        conn = await get_db_connection()
        print("Connection successful.")
    except Exception as e:
        print(f"Connection failed: {e}")
    finally:
        await close_db_connection(conn)


    print("\nFetching schema...")
    try:
        schema = await get_schema_description()
        print(schema)
    except Exception as e:
        print(f"Schema fetch failed: {e}")


    print("\nTesting query execution (example: first 5 clubs)...")
    try:
        # Example using named parameters (dictionary)
        clubs = await execute_query("SELECT clubId, clubName FROM clubs LIMIT :limit;", {"limit": 5})
        print(clubs)
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    import asyncio
    print("Running database test function. Ensure FEYOD_DATABASE_URL is set in your environment.")
    asyncio.run(_test_main())