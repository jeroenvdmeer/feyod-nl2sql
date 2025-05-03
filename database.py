"""Database connection and interaction logic using SQLAlchemy for async SQLite."""

import logging
from typing import List, Dict, Any, Optional
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
        if not config.DATABASE_URL:
            logger.error("DATABASE_URL is not configured. Cannot create database engine.")
            raise ValueError("Database URL is not configured.")
        try:
            logger.info(f"Creating async SQLAlchemy engine for URL: {config.DATABASE_URL}")
            _engine = create_async_engine(config.DATABASE_URL)
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
    except Exception as e: # Catch other potential errors like invalid URL format during connect
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
        # Use SQLAlchemy's reflection/introspection capabilities if needed for more complex scenarios
        # For SQLite, PRAGMA statements are often sufficient and simpler.

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
            col_result = await conn.execute(text(f"PRAGMA table_info('{table_name}');")) # Use quotes for safety
            columns = col_result.fetchall()
            for column in columns:
                # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
                col_name = column[1]
                col_type = column[2]
                col_pk = " (Primary Key)" if column[5] else ""
                schema_parts.append(f"  - {col_name}: {col_type}{col_pk}")
            schema_parts.append("") # Add a blank line between tables

        return "\n".join(schema_parts)

    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error retrieving schema: {e}")
        return f"Error retrieving schema: {e}"
    except Exception as e:
        logger.error(f"Unexpected error retrieving schema: {e}")
        return f"Unexpected error retrieving schema: {e}"
    finally:
        await close_db_connection(conn)


async def execute_query(sql: str, params: Optional[Dict | tuple] = None) -> List[Dict[str, Any]]:
    """Executes a given SQL query safely with parameters and returns results as dicts."""
    conn: Optional[AsyncConnection] = None
    results: List[Dict[str, Any]] = []
    if params is None:
        params = {} # Use empty dict for execute if no params provided

    try:
        conn = await get_db_connection()
        logger.info(f"Executing SQL: {sql} with params: {params}")
        # Use text() for literal SQL; ensure parameters are passed correctly
        query = text(sql)
        result_proxy = await conn.execute(query, params)

        # Check if the query returns rows (e.g., SELECT)
        if result_proxy.returns_rows:
            # Fetch all rows and convert them to dictionaries
            rows = result_proxy.mappings().fetchall() # .mappings() gives dict-like objects
            results = [dict(row) for row in rows]
            logger.info(f"Query executed successfully. Fetched {len(results)} rows.")
        else:
            # For non-SELECT statements (though we aim to only allow SELECTs)
            logger.info(f"Query executed successfully. Rows affected: {result_proxy.rowcount}")
            # For INSERT/UPDATE/DELETE, you might want conn.commit() if autocommit isn't enabled
            # await conn.commit() # Typically needed if not using autocommit

        # Consider adding commit for safety, though SELECTs don't need it and autocommit might be default
        # await conn.commit()

    except sqlalchemy.exc.SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error executing query '{sql}' with params {params}: {e}")
        # Propagate a clear error message
        raise ValueError(f"Error executing query: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error executing query '{sql}': {e}")
        raise ValueError(f"Unexpected error executing query: {e}")
    finally:
        await close_db_connection(conn)
    return results

# Example usage (for testing purposes) - Requires running in an async context
async def _test_main():
    logging.basicConfig(level=logging.INFO)
    # Set DATABASE_URL environment variable before running for this test
    if not config.DATABASE_URL:
        print("Please set the DATABASE_URL environment variable to run the test.")
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
        # Example using positional parameters (tuple) - less common with SQLAlchemy text()
        # clubs = await execute_query("SELECT clubId, clubName FROM clubs LIMIT ?;", (5,)) # Check DB driver support
        print(clubs)
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    import asyncio
    # Note: Running this directly might require setting env vars beforehand
    # or modifying config loading for testing.
    print("Running database test function. Ensure DATABASE_URL is set in your environment.")
    asyncio.run(_test_main())

