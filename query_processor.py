"""Handles the core logic for SQL generation, checking, and fixing."""

import logging
import aiosqlite # Keep for check_sql_syntax if not using SQLAlchemy's check
from typing import Tuple, Optional

# LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.output_parsers.string import StrOutputParser
except ImportError:
    logging.warning("langchain_core not found, falling back to older langchain imports.")
    from langchain.prompts import ChatPromptTemplate # type: ignore
    from langchain.schema.runnable import Runnable # type: ignore
    from langchain.chat_models.base import BaseChatModel # type: ignore
    from langchain.schema import StrOutputParser # type: ignore


# Local imports (relative)
from . import database
from .llm_factory import get_llm
from .examples import get_few_shot_prompt_template # Import the prompt template getter

logger = logging.getLogger(__name__)

# --- Prompt Definitions ---

# Define the system message content separately for clarity
SQL_GENERATION_SYSTEM_PROMPT = """
You are an expert SQLite assistant with strong attention to detail. Given the question, database table schema, and example queries, output a valid SQLite query. When generating the query, follow these rules:

**Core Logic & Context:**
- The input question is likely from the perspective of a fan of the football club Feyenoord. Use this knowledge when generating a query. For example, when data about a football match is requested and only an opponent is mentioned, assume that the other club is Feyenoord.
- When a club name is referenced, do not just use the columns homeClubName and awayClubName in the WHERE statement. Be smart, and also query the clubName column in the clubs table using the clubId. Additionally, take into account that a typo can have been made in the club name, so make the query flexible (e.g., using LIKE or checking variations if appropriate, but prioritize joining with the `clubs` table via `clubId`).
- When dates are mentioned in the question, remember to use the `strftime` function for comparisons if the database stores dates as text or numbers in a specific format. Assume dates are stored in 'YYYY-MM-DD HH:MM:SS' format unless schema indicates otherwise.

**Query Structure & Best Practices:**
- Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results. You can order the results by a relevant column to return the most interesting examples in the database.
- Never query for all the columns from a specific table (e.g., `SELECT *`). Only select the specific columns relevant to the question.
- DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Only SELECT statements are allowed.
- Double-check for common mistakes:
    - Using `NOT IN` with subqueries that might return NULL values.
    - Using `UNION` when `UNION ALL` is sufficient and more performant (if duplicates are acceptable or impossible).
    - Using `BETWEEN` for ranges; ensure inclusivity/exclusivity matches the intent.
    - Data type mismatches in predicates (e.g., comparing text to numbers).
    - Properly quoting identifiers (table names, column names) if they contain spaces or reserved keywords, although standard SQL identifiers usually don't require quotes in SQLite.
    - Using the correct number of arguments for SQL functions.
    - Casting data types explicitly if needed for comparisons or functions.
    - Ensuring correct join conditions, especially when joining multiple tables. Use the `clubs` table to select clubs based on `clubId` instead of relying solely on `homeClubName` and `awayClubName` text columns in other tables like `matches`.
    - Correct usage and placement of parentheses in complex `WHERE` clauses.

**Output Format:**
- Only output the raw SQL query. Do not include explanations, markdown formatting (like ```sql ... ```), or any text other than the SQL query itself.
"""

SQL_FIXING_SYSTEM_PROMPT = """
You are an expert SQLite assistant. You are given an invalid SQLite query, the error message it produced, the database schema, and the original natural language query.
Your task is to fix the SQL query so it is syntactically correct and likely addresses the user's original intent based on the provided context.

Database Schema:
{schema}

Original Natural Language Query:
{original_nl_query}

Invalid SQL Query:
{invalid_sql}

Syntax Error:
{error_message}

Rules for Fixing:
- Analyze the error message and the invalid SQL to understand the cause of the syntax error.
- Refer to the database schema to ensure table and column names are correct and used appropriately.
- Consider the original natural language query to maintain the intended logic.
- Apply SQLite syntax rules correctly. Pay attention to function usage, join conditions, quoting, data types, and clause structure.
- Only output the corrected, raw SQL query. Do not include explanations or markdown formatting.
"""

# --- Chain Builders ---

def _build_sql_generation_chain() -> Optional[Runnable]:
    """Builds the SQL generation chain with few-shot examples if available."""
    llm: Optional[BaseChatModel] = get_llm()
    if not llm:
        logger.error("LLM not available from factory. Cannot build SQL generation chain.")
        return None

    prompt_messages = [
        ("system", SQL_GENERATION_SYSTEM_PROMPT),
    ]

    # Get the few-shot prompt template (which includes the selector)
    few_shot_prompt = get_few_shot_prompt_template()
    if few_shot_prompt:
        logger.info("Adding few-shot examples to SQL generation prompt.")
        prompt_messages.append(few_shot_prompt)
    else:
        logger.warning("Few-shot examples not available, SQL generation prompt will not include them.")

    # Add the final human message template
    prompt_messages.append(("human", "=== Question:\n{natural_language_query}\n=== Schemas:\n{schema}\n=== Resulting query:"))

    prompt_template = ChatPromptTemplate.from_messages(prompt_messages)

    # Chain: Prompt -> LLM -> String Output Parser
    chain = prompt_template | llm | StrOutputParser()
    return chain

def _build_sql_fixing_chain() -> Optional[Runnable]:
    """Builds the SQL fixing chain."""
    llm: Optional[BaseChatModel] = get_llm()
    if not llm:
        logger.error("LLM not available from factory. Cannot build SQL fixing chain.")
        return None

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SQL_FIXING_SYSTEM_PROMPT),
        # Human message is implicitly handled by the input dictionary keys matching the system prompt template
    ])

    # Chain: Prompt -> LLM -> String Output Parser
    chain = prompt_template | llm | StrOutputParser()
    return chain


# --- Core Functions ---

async def generate_sql_from_nl(natural_language_query: str, schema: str) -> str:
    """Converts natural language query to SQL using the generation chain."""
    sql_generation_chain = _build_sql_generation_chain()
    if not sql_generation_chain:
         raise ValueError("SQL generation chain could not be built (LLM likely unavailable).")

    logger.info(f"Generating SQL for: {natural_language_query}")
    try:
        # Invoke the chain asynchronously
        sql_query = await sql_generation_chain.ainvoke({
            "schema": schema,
            "natural_language_query": natural_language_query
        })

        # Basic cleanup (LLM should follow instructions, but just in case)
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        sql_query = sql_query.strip().replace('\n', ' ') # Remove escaped newlines

        logger.info(f"Generated SQL: {sql_query}")
        if not sql_query or not sql_query.upper().startswith("SELECT"):
             # Log the problematic output for debugging
             logger.error(f"LLM did not return a valid SELECT query. Output: '{sql_query}'")
             raise ValueError("Generated query is not a valid SELECT statement.")
        return sql_query
    except Exception as e:
        logger.exception(f"Error invoking SQL generation chain: {e}")
        raise ValueError(f"Failed to generate SQL: {e}")


async def check_sql_syntax(sql_query: str) -> Tuple[bool, Optional[str]]:
    """
    Checks the syntax of an SQLite query using EXPLAIN.
    Returns (True, None) on success, (False, error_message) on failure.
    Uses a direct aiosqlite connection for this specific check, as it's simple
    and avoids potential complexities of checking via SQLAlchemy engine for just EXPLAIN.
    Requires DATABASE_URL to be a file path for aiosqlite.
    If DATABASE_URL is not a file path, this needs adjustment or use SQLAlchemy execute.
    """
    # Alternative: Use SQLAlchemy connection if DATABASE_URL isn't a direct file path
    # conn = None
    # try:
    #     conn = await database.get_db_connection()
    #     await conn.execute(text(f"EXPLAIN {sql_query}")) # Use text() with SQLAlchemy
    #     logger.info(f"SQL syntax check passed for: {sql_query}")
    #     return True, None
    # except database.sqlalchemy.exc.SQLAlchemyError as e:
    #     error_message = str(e)
    #     logger.warning(f"SQL syntax check failed for query '{sql_query}': {error_message}")
    #     return False, error_message
    # finally:
    #     await database.close_db_connection(conn)

    # --- Using direct aiosqlite (assuming file path in DATABASE_URL) ---
    conn = None
    db_path = database.config.DATABASE_URL
    if not db_path or not db_path.startswith("sqlite+aiosqlite:///"):
        logger.error("Cannot perform direct aiosqlite syntax check: DATABASE_URL is not a valid sqlite+aiosqlite path.")
        # Fallback or raise error - for now, assume valid syntax if URL is wrong format for this check
        # return True, "Warning: Could not perform syntax check due to DB URL format."
        # Or attempt SQLAlchemy check here as shown above.
        # Let's raise an error for clarity.
        raise ValueError("DATABASE_URL format not suitable for direct aiosqlite syntax check.")

    # Extract file path
    db_file_path = db_path[len("sqlite+aiosqlite:///"):]

    try:
        # Use a temporary connection just for the syntax check
        conn = await aiosqlite.connect(db_file_path)
        await conn.execute(f"EXPLAIN {sql_query}")
        logger.info(f"SQL syntax check passed for: {sql_query}")
        return True, None
    except aiosqlite.Error as e:
        error_message = str(e)
        logger.warning(f"SQL syntax check failed for query '{sql_query}': {error_message}")
        return False, error_message
    except FileNotFoundError:
        logger.error(f"Database file not found at {db_file_path} for syntax check.")
        return False, f"Database file not found at {db_file_path}"
    except Exception as e:
        logger.exception(f"Unexpected error during SQL syntax check: {e}")
        return False, f"Unexpected error during syntax check: {e}"
    finally:
        if conn:
            await conn.close()
    # --- End aiosqlite section ---


async def attempt_fix_sql(invalid_sql: str, error_message: str, schema: str, original_nl_query: str) -> str:
    """Attempts to fix an invalid SQL query using the fixing chain."""
    sql_fixing_chain = _build_sql_fixing_chain()
    if not sql_fixing_chain:
        raise ValueError("SQL fixing chain could not be built (LLM likely unavailable).")

    logger.warning(f"Attempting to fix SQL: {invalid_sql} based on error: {error_message}")
    try:
        # Invoke the chain asynchronously
        fixed_sql = await sql_fixing_chain.ainvoke({
            "schema": schema,
            "original_nl_query": original_nl_query,
            "invalid_sql": invalid_sql,
            "error_message": error_message
        })

        # Basic cleanup
        fixed_sql = fixed_sql.strip()
        if fixed_sql.startswith("```sql"):
            fixed_sql = fixed_sql[6:]
        if fixed_sql.endswith("```"):
            fixed_sql = fixed_sql[:-3]
        fixed_sql = fixed_sql.strip().replace('\n', ' ') # Remove escaped newlines

        logger.info(f"Proposed fixed SQL: {fixed_sql}")
        if not fixed_sql or not fixed_sql.upper().startswith("SELECT"):
            # Log the problematic output for debugging
            logger.error(f"LLM did not return a valid fixed SELECT query. Output: '{fixed_sql}'")
            raise ValueError("Fix attempt did not return a valid SELECT statement.")
        return fixed_sql
    except Exception as e:
        logger.exception(f"Error invoking SQL fixing chain: {e}")
        raise ValueError(f"Failed to fix SQL: {e}")

