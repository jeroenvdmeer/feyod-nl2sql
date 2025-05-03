# feyod-common

This package provides shared, reusable Python components for the [`feyod`](https://www.feyod.nl/) project ecosystem. It aims to centralize common functionalities like configuration, database access, and language model interactions. The components are used in [`feyod-mcp`](https://github.com/jeroenvdmeer/feyod-mcp) and [`feyod-chatbot-web`](https://github.com/jeroenvdmeer/feyod-chatbot-web).

## Components

*   **`config.py`**: Manages loading configuration settings (e.g., database URLs, API keys) from environment variables using `python-dotenv`.
*   **`database.py`**: Offers asynchronous database interaction capabilities, primarily designed for SQLite using `SQLAlchemy`'s async features. It includes utilities for establishing connections, retrieving schema information, and executing SQL queries safely.
*   **`llm_factory.py`**: A factory module for creating instances of Language Models (LLMs) and Embeddings based on configured providers (e.g., OpenAI, Google). It leverages the `LangChain` library.
*   **`query_processor.py`**: Implements the logic for converting natural language questions into SQL queries. It uses the configured LLM, database schema information, and potentially few-shot examples to generate, validate, and attempt to fix SQL queries.
*   **`examples.py`**: Handles loading and managing few-shot examples used to improve the accuracy of the natural language to SQL generation process.
*   **`utils/`**: Contains miscellaneous utility functions.
*   **`tests/`**: Includes tests for the common components.

## Database Setup

The components in this library, particularly `database.py` and `query_processor.py`, rely on a SQLite database (`feyod.db`) containing Feyenoord match data. This database is maintained in the main [feyod repository](https://github.com/jeroenvdmeer/feyod).

To set up the database:

1.  **Clone the main `feyod` repository (if you haven't already):**
    ```bash
    git clone https://github.com/jeroenvdmeer/feyod.git
    ```
2.  **Navigate to the `feyod` directory:**
    ```bash
    cd feyod
    ```
3.  **Build the SQLite database using the provided SQL file:**
    ```bash
    sqlite3 feyod.db < feyod.sql
    ```
    This command creates the `feyod.db` file in the root of the `feyod` repository.

4.  **Configure the `FEYOD_DATABASE_URL`:** Ensure the `FEYOD_DATABASE_URL` environment variable points to the location of this `feyod.db` file. For example, if your project using `feyod-common` is in a sibling directory to `feyod`, you might set:
    ```dotenv
    FEYOD_DATABASE_URL="sqlite+aiosqlite:///../feyod/feyod.db"
    ```

## Configuration

Before using the components in this package, ensure the necessary environment variables are set. These variables configure aspects like database connections and LLM providers/API keys. Refer to `config.py` for a detailed list of required and optional environment variables. Common variables include:

*   `FEYOD_DATABASE_URL`: The connection string for the database (e.g., `sqlite+aiosqlite:///./feyod.db`). **See Database Setup section.**
*   `LLM_PROVIDER`: The desired language model provider (e.g., `openai`, `google`).
*   `LLM_API_KEY`: The API key for the selected LLM provider.
*   `LLM_MODEL`: The specific model to use (e.g., `o4-mini`).
*   `EXAMPLE_SOURCE`: Where to load few-shot examples from (e.g., `local`, `mongodb`).

Create a `.env` file in the root of your project using these components and define the variables there.

## Usage

Modules within this package are intended to be imported and used by other parts of the `feyod` project. Ensure necessary environment variables are set as defined in `config.py` before using components that rely on them.