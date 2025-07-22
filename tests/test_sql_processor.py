import pytest
from unittest.mock import patch, AsyncMock

from feyod_nl2sql.workflow import sql_processor
from nl2sql.src.workflow import database

TEST_DB_URL = 'sqlite+aiosqlite:///./test_feyod_sql.sqlite'

@pytest.fixture
def mock_llm_chain():
    """Fixture to provide a mocked LLM chain."""
    chain = AsyncMock()
    chain.ainvoke = AsyncMock()
    return chain

@pytest.mark.asyncio
@patch('feyod_nl2sql.workflow.sql_processor._build_sql_generation_chain')
async def test_generate_sql_from_nl_success(mock_build_chain, mock_llm_chain):
    """Test successful SQL generation."""
    mock_build_chain.return_value = mock_llm_chain
    mock_llm_chain.ainvoke.return_value = "SELECT name FROM test_table LIMIT 5;"
    
    nl_query = "show me names"
    schema = "Table 'test_table'"
    messages = []

    result = await sql_processor.generate_sql_from_nl(nl_query, schema, messages)
    
    assert result == "SELECT name FROM test_table LIMIT 5;"
    mock_llm_chain.ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('feyod_nl2sql.workflow.sql_processor._build_sql_generation_chain')
async def test_generate_sql_from_nl_failure(mock_build_chain, mock_llm_chain):
    """Test failure in SQL generation chain."""
    mock_build_chain.return_value = mock_llm_chain
    mock_llm_chain.ainvoke.side_effect = Exception("LLM Error")

    with pytest.raises(ValueError, match="Failed to generate SQL"):
        await sql_processor.generate_sql_from_nl("q", "s", [])

@pytest.mark.asyncio
@patch('aiosqlite.connect', new_callable=AsyncMock)
async def test_check_sql_syntax_valid(mock_connect):
    """Test valid SQL syntax check."""
    mock_conn = AsyncMock()
    mock_connect.return_value = mock_conn
    
    is_valid, error = await database.check_sql_syntax("SELECT * FROM test", TEST_DB_URL)
    
    assert is_valid is True
    assert error is None

@pytest.mark.asyncio
@patch('aiosqlite.connect', new_callable=AsyncMock)
async def test_check_sql_syntax_invalid(mock_connect):
    """Test invalid SQL syntax check."""
    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = Exception("Syntax Error")
    mock_connect.return_value = mock_conn

    is_valid, error = await database.check_sql_syntax("SELEC * FROOM test", TEST_DB_URL)
    
    assert is_valid is False
    assert "Syntax Error" in error

@pytest.mark.asyncio
@patch('feyod_nl2sql.workflow.sql_processor._build_sql_fixing_chain')
async def test_attempt_fix_sql_success(mock_build_chain, mock_llm_chain):
    """Test successful SQL fixing."""
    mock_build_chain.return_value = mock_llm_chain
    mock_llm_chain.ainvoke.return_value = "SELECT * FROM test;"
    
    result = await sql_processor.attempt_fix_sql("invalid", "error", "schema", "query")
    
    assert result == "SELECT * FROM test;"
    mock_llm_chain.ainvoke.assert_called_once()

