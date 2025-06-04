"""Configuration settings for shared components, loaded from environment variables."""

import os
from dotenv import load_dotenv
import logging

# Load environment variables from a .env file in the consuming project
load_dotenv()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Default to INFO level

MAX_FIX_ATTEMPTS = int(os.getenv("MAX_FIX_ATTEMPTS", 3)) # Default to 3 attempts

# Database Configuration
# Use FEYOD_DATABASE_URL for flexibility (e.g., "sqlite+aiosqlite:///path/to/db.sqlite")
FEYOD_DATABASE_URL = os.getenv("FEYOD_DATABASE_URL")
if not FEYOD_DATABASE_URL:
    logger.warning("FEYOD_DATABASE_URL environment variable not set.")
    # Provide a default or raise an error if critical

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower() # Default to openai
LLM_API_KEY = os.getenv("LLM_API_KEY") # Key for the primary provider
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-04-17") # Default model

# Example Loading Configuration
EXAMPLE_SOURCE = os.getenv("EXAMPLE_SOURCE", "local").lower() # "local" or "mongodb"
EXAMPLE_DB_CONNECTION_STRING = os.getenv("EXAMPLE_DB_CONNECTION_STRING")
EXAMPLE_DB_NAME = os.getenv("EXAMPLE_DB_NAME", "feyenoord_data") # Default DB name
EXAMPLE_DB_COLLECTION = os.getenv("EXAMPLE_DB_COLLECTION", "examples") # Default collection name

# Context window management (memory)
CONTEXT_RECENT_MESSAGES_KEPT = int(os.getenv("CONTEXT_RECENT_MESSAGES_KEPT", 15))
CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD = int(os.getenv("CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD", 3000))

# Validate essential configurations
if not LLM_API_KEY and LLM_PROVIDER != "mock": # Allow mock provider without key
    logger.warning(f"LLM_API_KEY not set for provider '{LLM_PROVIDER}'. LLM/Embeddings may fail.")

if EXAMPLE_SOURCE == "mongodb" and not EXAMPLE_DB_CONNECTION_STRING:
    logger.warning("EXAMPLE_SOURCE is 'mongodb' but EXAMPLE_DB_CONNECTION_STRING is not set.")

