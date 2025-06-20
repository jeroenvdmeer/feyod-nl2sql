"""
Centralized configuration management for the feyod-nl2sql package.

This module loads all necessary environment variables from a .env file
and makes them available as constants for the rest of the application.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from a .env file.
# This allows for easy configuration in different environments.
load_dotenv()

logger = logging.getLogger(__name__)

# --- General Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HOST = os.getenv("HOST", "127.0.0.1")

# --- Workflow Configuration ---
MAX_SQL_FIX_ATTEMPTS = int(os.getenv("MAX_SQL_FIX_ATTEMPTS", 1))

# --- Database Configuration ---
FEYOD_DATABASE_URL = os.getenv("FEYOD_DATABASE_URL")
if not FEYOD_DATABASE_URL:
    logger.warning("FEYOD_DATABASE_URL environment variable is not set. Database operations will fail.")

# --- LLM Configuration ---
# For clarity, we can allow specific keys or a general one.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable is not set. LLM calls will fail.")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower() # Default to openai
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
if EXAMPLE_SOURCE == "mongodb" and not EXAMPLE_DB_CONNECTION_STRING:
    logger.warning("EXAMPLE_SOURCE is 'mongodb' but EXAMPLE_DB_CONNECTION_STRING is not set.")

