"""Factory for creating LLM and Embeddings instances based on configuration."""

import logging
from typing import Dict, Type, Optional

# Use BaseChatModel and Embeddings from langchain_core if possible, else langchain
try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.embeddings import Embeddings as BaseEmbeddings
except ImportError:
    logging.warning("langchain_core not found, falling back to older langchain imports.")
    from langchain.chat_models.base import BaseChatModel # type: ignore
    from langchain.embeddings.base import Embeddings as BaseEmbeddings # type: ignore


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# Import other provider classes here as needed
# from langchain_anthropic import ChatAnthropic

# Import config from the common module
from . import config

logger = logging.getLogger(__name__)

# Provider registry mapping provider names to their classes and required config keys
PROVIDER_REGISTRY: Dict[str, Dict[str, Type | str | Dict]] = {
    "openai": {
        "llm": ChatOpenAI,
        "embeddings": OpenAIEmbeddings,
        "api_key_config": "LLM_API_KEY",
        "llm_args": {"model_name": config.LLM_MODEL},
        "embeddings_args": {}, # Default args for OpenAI Embeddings
    },
    "google": {
        "llm": ChatGoogleGenerativeAI,
        "embeddings": GoogleGenerativeAIEmbeddings,
        "api_key_config": "GOOGLE_API_KEY", # Use specific key if available
        "llm_args": {"model": config.LLM_MODEL},
        "embeddings_args": {"model": "models/embedding-001"}, # Default Google embedding model
    },
    # Add other providers here, e.g.:
    # "anthropic": {
    #     "llm": ChatAnthropic,
    #     "embeddings": None, # Anthropic might not have dedicated embeddings class via langchain
    #     "api_key_config": "ANTHROPIC_API_KEY",
    #     "llm_args": {"temperature": 0.2, "model_name": config.LLM_MODEL},
    #     "embeddings_args": {},
    # },
}

def _get_api_key(provider: str) -> Optional[str]:
    """Gets the API key for the given provider from config."""
    provider_info = PROVIDER_REGISTRY.get(provider)
    if not provider_info:
        return None

    api_key_config_name = provider_info.get("api_key_config")
    if not isinstance(api_key_config_name, str):
         logger.warning(f"API key config name not found or invalid for provider '{provider}'.")
         return None

    api_key = getattr(config, api_key_config_name, None)

    # Fallback for Google if specific key not found but primary LLM_API_KEY exists
    if provider == "google" and not api_key and config.LLM_API_KEY:
        api_key = config.LLM_API_KEY
        logger.debug("Using LLM_API_KEY as fallback for Google API key.")

    if not api_key:
         logger.warning(f"API key ('{api_key_config_name}' or fallback) not found in config for provider '{provider}'.")

    return api_key


def get_llm() -> Optional[BaseChatModel]:
    """Creates an LLM instance based on the configured provider."""
    provider = config.LLM_PROVIDER
    logger.info(f"Attempting to initialize LLM for provider: {provider}")

    if provider not in PROVIDER_REGISTRY:
        logger.error(f"Unsupported LLM_PROVIDER: {provider}")
        return None

    provider_info = PROVIDER_REGISTRY[provider]
    llm_class = provider_info.get("llm")
    if not llm_class or not isinstance(llm_class, type):
        logger.error(f"LLM class not defined or invalid for provider: {provider}")
        return None

    api_key = _get_api_key(provider)
    if not api_key:
        logger.error(f"Cannot initialize LLM for provider {provider} due to missing API key.")
        return None

    # Prepare arguments
    constructor_args = provider_info.get("llm_args", {})
    if not isinstance(constructor_args, dict):
        logger.error(f"Invalid llm_args format for provider {provider}")
        return None

    # Add API key based on provider's expected parameter name
    # This needs adjustment based on actual class signatures if adding more providers
    if provider == "openai":
        constructor_args["api_key"] = api_key
        # Ensure model_name is passed correctly, overriding default if needed
        constructor_args["model_name"] = config.LLM_MODEL
    elif provider == "google":
        constructor_args["google_api_key"] = api_key
        # Ensure model is passed correctly
        constructor_args["model"] = config.LLM_MODEL
    # Add elif for other providers' key names

    try:
        logger.info(f"Initializing {provider} LLM (Model: {config.LLM_MODEL}). Args: {constructor_args}")
        llm_instance = llm_class(**constructor_args)
        logger.info(f"LLM initialized successfully for provider: {provider}")
        return llm_instance
    except Exception as e:
        logger.exception(f"Failed to initialize LLM for provider {provider}: {e}")
        return None


def get_embeddings() -> Optional[BaseEmbeddings]:
    """Creates an Embeddings instance based on the configured provider."""
    provider = config.LLM_PROVIDER # Assume embeddings provider matches LLM provider
    logger.info(f"Attempting to initialize Embeddings for provider: {provider}")

    if provider not in PROVIDER_REGISTRY:
        logger.error(f"Unsupported provider for embeddings: {provider}")
        return None

    provider_info = PROVIDER_REGISTRY[provider]
    embeddings_class = provider_info.get("embeddings")

    if not embeddings_class or not isinstance(embeddings_class, type):
        logger.warning(f"Embeddings class not defined, invalid, or not applicable for provider: {provider}. Cannot initialize embeddings.")
        return None

    api_key = _get_api_key(provider)
    if not api_key:
        logger.error(f"Cannot initialize Embeddings for provider {provider} due to missing API key.")
        return None

    embeddings_args = provider_info.get("embeddings_args", {})
    if not isinstance(embeddings_args, dict):
        logger.error(f"Invalid embeddings_args format for provider {provider}")
        return None

    # Prepare arguments
    constructor_args = {**embeddings_args}
    # Add API key based on provider's expected parameter name
    if provider == "openai":
        constructor_args["api_key"] = api_key
    elif provider == "google":
         constructor_args["google_api_key"] = api_key
         # Ensure model is set if not already in defaults
         constructor_args.setdefault("model", "models/embedding-001")
    # Add elif for other providers' key names

    try:
        logger.info(f"Initializing {provider} Embeddings. Args: {constructor_args}")
        embeddings_instance = embeddings_class(**constructor_args)
        logger.info(f"Embeddings initialized successfully for provider: {provider}")
        return embeddings_instance
    except Exception as e:
        logger.exception(f"Failed to initialize Embeddings for provider {provider}: {e}")
        return None

