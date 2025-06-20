import pytest
import pytest_asyncio
import os
from unittest.mock import MagicMock
from sqlalchemy import create_engine, text

from feyod_nl2sql.workflow import database

TEST_DB_URL = 'sqlite+aiosqlite:///./test_feyod.sqlite'

@pytest.fixture(scope="module")
def setup_test_db():
    """Set up a test database and clean it up after tests."""
    sync_db_url = TEST_DB_URL.replace('+aiosqlite', '')
    engine = create_engine(sync_db_url)
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS clubs (clubId INTEGER PRIMARY KEY, clubName TEXT)"))
        conn.execute(text("INSERT INTO clubs (clubId, clubName) VALUES (1, 'Feyenoord')"))
        conn.execute(text("COMMIT"))
    
    yield
    
    db_path = TEST_DB_URL.replace('sqlite+aiosqlite:///', '')
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.mark.asyncio
async def test_get_schema_description(setup_test_db):
    """Test that the schema description can be retrieved correctly."""
    # Mock the LLM as it's a dependency for the toolkit
    mock_llm = MagicMock()
    
    schema = await database.get_schema_description(db_url=TEST_DB_URL, llm=mock_llm)
    
    assert "clubs" in schema
    assert "clubId" in schema
    assert "clubName" in schema

@pytest.mark.asyncio
async def test_execute_query(setup_test_db):
    """Test that a valid query can be executed."""
    results = await database.execute_query("SELECT clubName FROM clubs WHERE clubId = 1", db_url=TEST_DB_URL)
    assert len(results) == 1
    assert results[0]['clubName'] == 'Feyenoord'

@pytest.mark.asyncio
async def test_execute_query_no_results(setup_test_db):
    """Test that a query with no results returns an empty list."""
    results = await database.execute_query("SELECT clubName FROM clubs WHERE clubId = 99", db_url=TEST_DB_URL)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_execute_query_invalid_sql(setup_test_db):
    """Test that an invalid query raises a ValueError."""
    with pytest.raises(ValueError, match="Query execution failed"):
        await database.execute_query("SELECT * FROM non_existent_table", db_url=TEST_DB_URL)

@pytest.mark.asyncio
async def test_get_distinct_values(setup_test_db):
    """Test retrieving distinct values from a column."""
    club_names = await database.get_distinct_values("clubName", "clubs", db_url=TEST_DB_URL)
    assert isinstance(club_names, list)
    assert "Feyenoord" in club_names

