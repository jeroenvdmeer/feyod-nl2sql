import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Mock LLM and other dependencies before importing the module under test
mock_llm_instance = AsyncMock()
mock_llm_instance.ainvoke = AsyncMock()

mock_embeddings_instance = MagicMock()

mock_few_shot_template = MagicMock() # Simplified mock

# Patch the factory functions *before* importing query_processor
with patch('common.llm_factory.get_llm', return_value=mock_llm_instance), \
     patch('common.llm_factory.get_embeddings', return_value=mock_embeddings_instance), \
     patch('common.examples.get_few_shot_prompt_template', return_value=mock_few_shot_template):
    from .. import query_processor
    from .. import database # Need to patch database connection for check_sql_syntax

@pytest.fixture(autouse=True)
def reset_mocks():
    mock_llm_instance.ainvoke.reset_mock()

# --- Test generate_sql_from_nl ---

@pytest.mark.asyncio
async def test_generate_sql_from_nl_success():
    mock_llm_instance.ainvoke.return_value = "SELECT name FROM test_table LIMIT 5;"
    nl_query = "show me names"
    schema = "Table 'test_table':\n  - name: TEXT"

    result = await query_processor.generate_sql_from_nl(nl_query, schema)

    assert result == "SELECT name FROM test_table LIMIT 5;"
    mock_llm_instance.ainvoke.assert_called_once()
    call_args = mock_llm_instance.ainvoke.call_args[0][0]
    assert call_args['natural_language_query'] == nl_query
    assert call_args['schema'] == schema

@pytest.mark.asyncio
async def test_generate_sql_from_nl_llm_error():
    mock_llm_instance.ainvoke.side_effect = Exception("LLM API Error")
    nl_query = "show me names"
    schema = "Table 'test_table':\n  - name: TEXT"

    with pytest.raises(ValueError, match="Failed to generate SQL: LLM API Error"):
        await query_processor.generate_sql_from_nl(nl_query, schema)

@pytest.mark.asyncio
async def test_generate_sql_from_nl_invalid_output():
    mock_llm_instance.ainvoke.return_value = "Here is the query: SELECT name FROM test_table;" # Invalid format
    nl_query = "show me names"
    schema = "Table 'test_table':\n  - name: TEXT"

    with pytest.raises(ValueError, match="Generated query is not a valid SELECT statement."):
         await query_processor.generate_sql_from_nl(nl_query, schema)

# --- Test check_sql_syntax ---

# Mock the direct aiosqlite connection used in check_sql_syntax
@pytest.mark.asyncio
@patch('aiosqlite.connect', new_callable=AsyncMock)
async def test_check_sql_syntax_valid(mock_connect):
    # Mock the connection and cursor execute
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_cursor)
    mock_connect.return_value = mock_conn # Mock the context manager entry

    # Mock config to provide a valid path
    with patch('common.config.FEYOD_DATABASE_URL', 'sqlite+aiosqlite:///./dummy.db'):
        is_valid, error = await query_processor.check_sql_syntax("SELECT * FROM test_table")

    assert is_valid is True
    assert error is None
    mock_conn.execute.assert_called_once_with("EXPLAIN SELECT * FROM test_table")
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
@patch('aiosqlite.connect', new_callable=AsyncMock)
async def test_check_sql_syntax_invalid(mock_connect):
    mock_conn = AsyncMock()
    # Simulate an error during execute
    mock_conn.execute = AsyncMock(side_effect=aiosqlite.Error("Syntax Error near blah"))
    mock_connect.return_value = mock_conn

    with patch('common.config.FEYOD_DATABASE_URL', 'sqlite+aiosqlite:///./dummy.db'):
        is_valid, error = await query_processor.check_sql_syntax("SELECT FROM test_table")

    assert is_valid is False
    assert "Syntax Error near blah" in error
    mock_conn.execute.assert_called_once_with("EXPLAIN SELECT FROM test_table")
    mock_conn.close.assert_called_once()

@pytest.mark.asyncio
async def test_check_sql_syntax_bad_db_url():
     with patch('common.config.FEYOD_DATABASE_URL', 'not_a_sqlite_url'):
          with pytest.raises(ValueError, match="FEYOD_DATABASE_URL format not suitable"):
               await query_processor.check_sql_syntax("SELECT * FROM test_table")


# --- Test attempt_fix_sql ---

@pytest.mark.asyncio
async def test_attempt_fix_sql_success():
    mock_llm_instance.ainvoke.return_value = "SELECT name FROM test_table;" # The fixed query
    invalid_sql = "SELEC name FROM test_table"
    error_msg = "Syntax error near SELEC"
    schema = "Table 'test_table':\n  - name: TEXT"
    nl_query = "show names"

    result = await query_processor.attempt_fix_sql(invalid_sql, error_msg, schema, nl_query)

    assert result == "SELECT name FROM test_table;"
    mock_llm_instance.ainvoke.assert_called_once()
    call_args = mock_llm_instance.ainvoke.call_args[0][0]
    assert call_args['invalid_sql'] == invalid_sql
    assert call_args['error_message'] == error_msg
    assert call_args['schema'] == schema
    assert call_args['original_nl_query'] == nl_query

@pytest.mark.asyncio
async def test_attempt_fix_sql_llm_error():
    mock_llm_instance.ainvoke.side_effect = Exception("LLM Fix Error")
    # ... (setup args like above)

    with pytest.raises(ValueError, match="Failed to fix SQL: LLM Fix Error"):
        await query_processor.attempt_fix_sql("invalid", "error", "schema", "query")

@pytest.mark.asyncio
async def test_attempt_fix_sql_invalid_output():
    mock_llm_instance.ainvoke.return_value = "I fixed it: SELECT name FROM test_table;" # Invalid format
    # ... (setup args like above)

    with pytest.raises(ValueError, match="Fix attempt did not return a valid SELECT statement."):
        await query_processor.attempt_fix_sql("invalid", "error", "schema", "query")

