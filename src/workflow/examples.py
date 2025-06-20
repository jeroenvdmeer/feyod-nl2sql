"""Sets up few-shot examples for SQL generation using semantic similarity."""

import logging
from typing import Optional, List, Dict, Any

# LangChain core imports
try:
    from langchain_core.embeddings import Embeddings as BaseEmbeddings
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector, BaseExampleSelector
    from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
    from langchain_core.vectorstores import VectorStore
except ImportError:
    logging.warning("langchain_core not found, falling back to older langchain imports.")
    # Fallback imports (adjust as needed based on specific langchain version)
    from langchain.embeddings.base import Embeddings as BaseEmbeddings # type: ignore
    from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate # type: ignore
    from langchain.schema import BaseExampleSelector # type: ignore
    from langchain.vectorstores.base import VectorStore # type: ignore


# Specific Vector Store implementation
from langchain_community.vectorstores import FAISS

# Local imports
from . import config
from .llm_factory import get_embeddings # Import the factory function

# Attempt to import pymongo, but don't fail if it's not needed/installed
try:
    import pymongo
    from pymongo.errors import ConfigurationError, ServerSelectionTimeoutError
except ImportError:
    pymongo = None
    ConfigurationError = None # type: ignore
    ServerSelectionTimeoutError = None # type: ignore


logger = logging.getLogger(__name__)

# --- Hardcoded Local Examples (Fallback) ---
# (Keep the examples from mcp/examples.py)
_local_examples = [
    {
        "id": "local-1", "natural_language_query": "Hoe vaak heeft Feyenoord gewonnen van Ajax?",
        "query": "SELECT COUNT(*) FROM matches WHERE (((homeClubName = 'Feyenoord' OR homeClubId = (SELECT clubId FROM clubs WHERE clubName='Feyenoord')) AND (awayClubName = 'Ajax' OR awayClubId = (SELECT clubId from clubs WHERE clubName='Ajax')) AND homeClubFinalScore > awayClubFinalScore) OR ((homeClubName = 'Ajax' OR homeClubId = (SELECT clubId FROM clubs WHERE clubName='Ajax')) AND (awayClubName = 'Feyenoord' OR awayClubId = (SELECT clubId FROM clubs WHERE clubName='Feyenoord')) AND awayClubFinalScore > homeClubFinalScore));"
    },
    {
        "id": "local-2", "natural_language_query": "Hoe vaak heeft hebben Coen Moulijn en Sjaak Swart tegelijk in een wedstrijd gescoord?",
        "query": "SELECT p1.playerName AS player1, p2.playerName AS player2, COUNT(DISTINCT g1.matchId) AS matches_together FROM goals g1 JOIN goals g2 ON g1.matchId = g2.matchId AND g1.playerId != g2.playerId JOIN players p1 ON g1.playerId = p1.playerId JOIN players p2 ON g2.playerId = p2.playerId WHERE (p1.playerName = 'Coen Moulijn' AND p2.playerName = 'Sjaak Swart')   OR (p1.playerName = 'Sjaak Swart' AND p2.playerName = 'Coen Moulijn') GROUP BY player1, player2;"
    },
    {
        "id": "local-3", "natural_language_query": "Wat is de grootste overwinning van Feyenoord op PSV?",
        "query": "SELECT m.dateAndTime, m.homeClubName, m.awayClubName, m.homeClubFinalScore, m.awayClubFinalScore FROM matches m WHERE (((homeClubName = 'Feyenoord' OR homeClubId = (SELECT clubId FROM clubs WHERE clubName='Feyenoord')) AND (awayClubName = 'PSV' OR awayClubId = (SELECT clubId from clubs WHERE clubName='PSV')) AND homeClubFinalScore > awayClubFinalScore) OR ((homeClubName = 'PSV' OR homeClubId = (SELECT clubId FROM clubs WHERE clubName='PSV')) AND (awayClubName = 'Feyenoord' OR awayClubId = (SELECT clubId FROM clubs WHERE clubName='Feyenoord')) AND awayClubFinalScore > homeClubFinalScore)) ORDER BY ABS(m.homeClubFinalScore - m.awayClubFinalScore) DESC, MAX(m.homeClubFinalScore, m.awayClubFinalScore) ASC, m.dateAndTime ASC LIMIT 5;"
    }
]


# --- Internal State Variables for Lazy Initialization ---
_examples: Optional[List[Dict[str, Any]]] = None
_embeddings_instance: Optional[BaseEmbeddings] = None
_vector_store: Optional[VectorStore] = None
_example_selector: Optional[BaseExampleSelector] = None
_few_shot_prompt_template: Optional[FewShotChatMessagePromptTemplate] = None

# --- Example Loading Logic ---

def _load_examples_from_mongodb() -> Optional[List[Dict[str, Any]]]:
    """Loads examples from MongoDB based on config."""
    if not pymongo or not ConfigurationError or not ServerSelectionTimeoutError:
        logger.error("MongoDB source selected, but 'pymongo' library is not installed. Cannot load examples from DB.")
        return None # Indicate failure

    if not config.EXAMPLE_DB_CONNECTION_STRING:
        logger.error("MongoDB source selected, but EXAMPLE_DB_CONNECTION_STRING is not set in .env. Cannot connect.")
        return None # Indicate failure

    client = None # Initialize client to None
    try:
        logger.info(f"Attempting to connect to MongoDB: {config.EXAMPLE_DB_NAME}/{config.EXAMPLE_DB_COLLECTION}")
        # Set a reasonable timeout
        client = pymongo.MongoClient(
            config.EXAMPLE_DB_CONNECTION_STRING,
            serverSelectionTimeoutMS=5000 # 5 seconds timeout
        )
        # The ismaster command is cheap and does not require auth. Forces connection check.
        client.admin.command('ismaster')
        logger.info("MongoDB connection successful.")

        db = client[config.EXAMPLE_DB_NAME]
        collection = db[config.EXAMPLE_DB_COLLECTION]

        # Fetch examples, ensuring required fields are present
        # Adjust projection if your documents have different field names
        fetched_examples = list(collection.find(
            {},
            {"_id": 0, "natural_language_query": 1, "query": 1} # Project only needed fields
        ))

        # Basic validation
        valid_examples = [
            ex for ex in fetched_examples
            if isinstance(ex.get("natural_language_query"), str) and isinstance(ex.get("query"), str)
        ]

        if not valid_examples:
             logger.warning(f"MongoDB query returned no valid examples from {config.EXAMPLE_DB_NAME}/{config.EXAMPLE_DB_COLLECTION}.")
             return [] # Return empty list, not None, as the query succeeded but found nothing

        logger.info(f"Successfully loaded {len(valid_examples)} examples from MongoDB.")
        # Add an 'id' if it's missing, useful for some LangChain components
        for i, ex in enumerate(valid_examples):
            ex.setdefault("id", f"db-{i+1}")
        return valid_examples

    except ConfigurationError as e:
        logger.error(f"MongoDB configuration error (check connection string?): {e}")
        return None # Indicate failure
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection timed out: {e}")
        return None # Indicate failure
    except Exception as e:
        logger.exception(f"An unexpected error occurred fetching examples from MongoDB: {e}")
        return None # Indicate failure
    finally:
        if client:
            client.close()
            logger.debug("MongoDB connection closed.")


def load_examples() -> List[Dict[str, Any]]:
    """Loads examples based on the configured source (config.EXAMPLE_SOURCE)."""
    global _examples
    if _examples is None: # Only load once
        source = config.EXAMPLE_SOURCE
        logger.info(f"Attempting to load examples from source: '{source}'")

        loaded_examples = None
        if source == 'mongodb':
            loaded_examples = _load_examples_from_mongodb()
            if loaded_examples is not None:
                logger.info(f"Using {len(loaded_examples)} examples loaded from MongoDB.")
            else:
                logger.warning("Failed to load examples from MongoDB. Falling back to local examples.")
                loaded_examples = _local_examples # Fallback to local
        elif source == 'local':
            logger.info(f"Using {len(_local_examples)} local hardcoded examples.")
            loaded_examples = _local_examples
        else:
            logger.warning(f"Unknown EXAMPLE_SOURCE '{source}' configured. Defaulting to local examples.")
            loaded_examples = _local_examples

        _examples = loaded_examples if loaded_examples is not None else [] # Ensure _examples is always a list

    return _examples


# --- Lazy Initializer Functions ---

def _get_embeddings_instance() -> Optional[BaseEmbeddings]:
    """Lazily initializes and returns the LangChain Embeddings using the factory."""
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info("Initializing Embeddings via factory (first access)...")
        _embeddings_instance = get_embeddings() # Call the factory function
        if _embeddings_instance:
            logger.info("Embeddings initialized successfully via factory.")
        else:
            logger.error("Failed to initialize Embeddings via factory.")
    return _embeddings_instance

def _get_vector_store() -> Optional[VectorStore]:
    """Lazily initializes and returns the FAISS Vector Store."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initializing FAISS Vector Store (first access)...")
        current_examples = load_examples()
        current_embeddings = _get_embeddings_instance()

        if current_embeddings and current_examples:
            try:
                logger.info(f"Creating FAISS vector store from {len(current_examples)} examples.")
                texts = [ex["natural_language_query"] for ex in current_examples]
                metadatas = current_examples
                _vector_store = FAISS.from_texts(
                    texts,
                    current_embeddings,
                    metadatas=metadatas
                )
                logger.info("FAISS vector store initialized successfully.")
            except Exception as e:
                logger.exception(f"Failed to create FAISS vector store: {e}. Few-shot examples will not use semantic similarity.")
                _vector_store = None # Ensure it's None on failure
        elif not current_examples:
            logger.warning("No examples loaded. Cannot create vector store.")
            _vector_store = None
        else: # Embeddings not available
            logger.warning("Embeddings not available. Cannot create vector store.")
            _vector_store = None
    return _vector_store


def get_few_shot_selector() -> Optional[BaseExampleSelector]:
    """
    Lazily initializes and returns the SemanticSimilarityExampleSelector.
    Returns None if examples, embeddings, or vector store cannot be initialized.
    """
    global _example_selector
    if _example_selector is None:
        logger.info("Initializing Few-Shot Example Selector (first access)...")
        vector_store = _get_vector_store()
        current_examples = load_examples() # Needed for k calculation

        if vector_store and current_examples:
            try:
                k = min(3, len(current_examples)) # Select up to 3 examples
                _example_selector = SemanticSimilarityExampleSelector(
                    vectorstore=vector_store,
                    k=k,
                    input_keys=["natural_language_query"], # Key in the input dict to use for similarity search
                )
                logger.info(f"SemanticSimilarityExampleSelector initialized successfully with k={k}.")
            except Exception as e:
                logger.exception(f"Failed to create SemanticSimilarityExampleSelector: {e}")
                _example_selector = None
        else:
            logger.warning("Vector store or examples not available. Cannot initialize example selector.")
            _example_selector = None

    return _example_selector


# Define the prompt template for formatting examples
# This is now separate from the selector itself
example_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("human", "{natural_language_query}"),
        ("ai", "{query}"), # 'query' is the key holding the SQL in our examples
    ]
)

def get_few_shot_prompt_template() -> Optional[FewShotChatMessagePromptTemplate]:
    """
    Lazily initializes and returns the FewShotChatMessagePromptTemplate.
    This combines the selector and the example formatting prompt.
    Returns None if the selector cannot be initialized.
    """
    global _few_shot_prompt_template
    if _few_shot_prompt_template is None:
        logger.info("Initializing FewShotChatMessagePromptTemplate (first access)...")
        selector = get_few_shot_selector()
        if selector:
            try:
                _few_shot_prompt_template = FewShotChatMessagePromptTemplate(
                    example_selector=selector,
                    example_prompt=example_prompt_template,
                    # The input variables for the *overall* prompt template that uses this few-shot template
                    input_variables=["natural_language_query"],
                )
                logger.info("FewShotChatMessagePromptTemplate initialized successfully.")
            except Exception as e:
                logger.exception(f"Failed to create FewShotChatMessagePromptTemplate: {e}")
                _few_shot_prompt_template = None
        else:
            logger.warning("Example selector not available. Cannot initialize FewShotChatMessagePromptTemplate.")
            _few_shot_prompt_template = None

    return _few_shot_prompt_template

