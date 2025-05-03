import pytest
import pytest_asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import os

# Set env var before importing config and database
# In a real scenario, use pytest-dotenv or fixtures to manage this
os.environ['DATABASE_URL'] = 'sqlite+aiosqlite:///./test_db.sqlite'

# Now import the modules under test
from .. import database
from .. import config

# Ensure the test database file is cleaned up
@pytest.fixture(scope="module", autouse=True)
def cleanup_db():
    yield
    db_path = config.DATABASE_URL[len('sqlite+aiosqlite:///'):]
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest_asyncio.fixture
async def db_conn():
    # Ensure engine is created for the test DB
    engine = database.get_engine()
    # Create a dummy table for testing schema/query execution
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: sync_conn.execute(
                "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
            )
        )
        await conn.run_sync(
             lambda sync_conn: sync_conn.execute(
                "INSERT INTO test_table (id, name) VALUES (1, 'test_item') ON CONFLICT(id) DO NOTHING"
             )
        )

    # Get a connection for the test function
    conn = await database.get_db_connection()
    yield conn
    await database.close_db_connection(conn)
    # Optional: Dispose engine after tests if needed, though usually not required per function
    # await engine.dispose()


@pytest.mark.asyncio
async def test_get_db_connection(db_conn):
    assert db_conn is not None
    assert hasattr(db_conn, 'execute') # Check if it looks like a connection object

@pytest.mark.asyncio
async def test_get_schema_description(db_conn):
    # We created 'test_table' in the fixture
    schema = await database.get_schema_description()
    assert "Table 'test_table':" in schema
    assert "- id: INTEGER (Primary Key)" in schema
    assert "- name: TEXT" in schema

@pytest.mark.asyncio
async def test_execute_query_select(db_conn):
    results = await database.execute_query("SELECT id, name FROM test_table WHERE id = :id", {"id": 1})
    assert len(results) == 1
    assert results[0]['id'] == 1
    assert results[0]['name'] == 'test_item'

@pytest.mark.asyncio
async def test_execute_query_no_results(db_conn):
    results = await database.execute_query("SELECT id, name FROM test_table WHERE id = :id", {"id": 99})
    assert len(results) == 0

@pytest.mark.asyncio
async def test_execute_query_invalid_sql(db_conn):
    with pytest.raises(ValueError, match="Error executing query"):
        await database.execute_query("SELECT * FROM non_existent_table")

# Test connection failure (more complex to mock engine/connect reliably)
# @pytest.mark.asyncio
# async def test_get_db_connection_failure():
#     with patch('common.database.create_async_engine', side_effect=Exception("Connection Refused")):
#         with pytest.raises(ValueError, match="Failed to connect to database"):
#              # Need to reset the global _engine or mock get_engine behavior
#              database._engine = None # Reset engine for this test
#              await database.get_db_connection()
#              database._engine = None # Clean up after

