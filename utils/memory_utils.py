import logging
from typing import List, Any
from langchain_core.messages import BaseMessage, SystemMessage
from common.llm_factory import get_llm
from common.config import CONTEXT_RECENT_MESSAGES_KEPT, CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD

logger = logging.getLogger(__name__)

def is_error_message(msg):
    return getattr(msg, "name", None) == "error"

def is_meaningful_message(msg):
    # Only include HumanMessage and AIMessage with a non-error name
    msg_type = getattr(msg, "type", None)
    if msg_type is None:
        # Fallback: check class name
        msg_type = msg.__class__.__name__.lower()
    return (msg_type in ("humanmessage", "aimessage")) and not is_error_message(msg)

async def _prepare_llm_context(full_messages: List[BaseMessage], config: Any = None) -> List[BaseMessage]:
    """
    Prepare a context window for the LLM by summarizing older messages if needed.
    Prioritizes user and AI answer messages in the recent buffer, only including error messages if needed.
    Summarizes older messages if their combined character count exceeds CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD.
    Collapses error chains in summary. Adds loop detection/clarification if needed.
    Returns a list of BaseMessage objects for LLM input.
    """
    logger.info(f"Preparing LLM context: {len(full_messages)} total messages.")
    logger.info(f"Config: CONTEXT_RECENT_MESSAGES_KEPT={CONTEXT_RECENT_MESSAGES_KEPT}, CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD={CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD}")
    if not full_messages:
        logger.info("No messages provided.")
        return []
    # --- Heuristic 1: Split into recent and older messages ---
    # Only the most recent CONTEXT_RECENT_MESSAGES_KEPT messages are considered 'recent'.
    # All earlier messages are 'older' and may be summarized if too verbose.
    if len(full_messages) > CONTEXT_RECENT_MESSAGES_KEPT:
        recent_messages = full_messages[-CONTEXT_RECENT_MESSAGES_KEPT:]
        older_messages = full_messages[:-CONTEXT_RECENT_MESSAGES_KEPT]
    else:
        recent_messages = full_messages[:]
        older_messages = []
    # --- Heuristic 2: Loop/failure detection ---
    # If the recent buffer is almost entirely error messages, prepend a clarification SystemMessage.
    # This helps break error loops and signals the LLM to ask for clarification.
    error_count = sum(1 for m in recent_messages if is_error_message(m))
    clarification_needed = error_count >= CONTEXT_RECENT_MESSAGES_KEPT - 1
    clarification = None
    if clarification_needed:
        clarification = SystemMessage(
            content="Er zijn meerdere fouten opgetreden. Kun je je vraag anders formuleren of verduidelijken?",
            name="clarification_request"
        )
        # Only keep the most recent errors to avoid flooding the context with redundant errors
        recent_messages = recent_messages[-(CONTEXT_RECENT_MESSAGES_KEPT-1):]
    # --- Heuristic 3: Summarize older messages if too verbose ---
    # If the combined character count of older messages exceeds the threshold, summarize them.
    older_char_count = sum(len(getattr(m, 'content', '')) for m in older_messages)
    logger.info(f"older_messages count: {len(older_messages)}, recent_messages count: {len(recent_messages)}, older_char_count: {older_char_count}")
    # --- Heuristic 4: Collapse error chains in summary ---
    # When summarizing, collapse consecutive error messages into a single summary line to avoid repetition.
    def collapse_errors(msgs):
        collapsed = []
        error_run = 0
        for m in msgs:
            if is_error_message(m):
                error_run += 1
            else:
                if error_run > 1:
                    collapsed.append(SystemMessage(content=f"({error_run} errors omitted)", name="error_summary"))
                elif error_run == 1:
                    collapsed.append(SystemMessage(content="(1 error omitted)", name="error_summary"))
                error_run = 0
                collapsed.append(m)
        if error_run > 1:
            collapsed.append(SystemMessage(content=f"({error_run} errors omitted)", name="error_summary"))
        elif error_run == 1:
            collapsed.append(SystemMessage(content="(1 error omitted)", name="error_summary"))
        return collapsed
    if older_messages and older_char_count > CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD:
        logger.info(f"Summarization triggered for {len(older_messages)} older messages with {older_char_count} characters.")
        llm = get_llm()
        # Collapse error chains before summarization to keep the summary concise
        older_collapsed = collapse_errors(older_messages)
        older_content = '\n'.join(getattr(m, 'content', '') for m in older_collapsed)
        summary_prompt = (
            "You are an expert at summarizing conversation histories. "
            "Condense the following messages into a brief, neutral summary that captures the key topics, decisions, and entities discussed. "
            "If there were repeated errors, summarize them as a single line. "
            "Retain essential information that would provide context for a continuing conversation.\n"
            "Here is the conversation history to summarize:\n"
            f"{older_content}"
        )
        logger.debug(f"Summarization prompt: {summary_prompt[:500]}...")
        try:
            summary_response = await llm.ainvoke(summary_prompt)
            summary_text = getattr(summary_response, 'content', str(summary_response))
            logger.info(f"Summary received: {summary_text[:300]}...")
            summary_message = SystemMessage(content=summary_text, name="conversation_summary")
            # Only summary is prepended (not clarification), then recent messages
            context = [summary_message]
            context += recent_messages
            logger.info(f"Prepared LLM context: {len(context)} messages (summary and recent messages).")
            logger.info(f"Context types: {[type(m) for m in context]}")
            return context
        except Exception as e:
            logger.exception("Error during LLM summarization. Falling back to truncation.")
            logger.info(f"Returning only recent_messages, count: {len(recent_messages)}")
            context = []
            if clarification_needed:
                context.append(clarification)
            context += recent_messages
            return context
    # --- Heuristic 5: Default context return ---
    # If no summarization is needed, return (clarification if needed) + all older + all recent messages
    logger.info("No summarization needed. Returning recent and older messages as context.")
    context = []
    if clarification_needed:
        context.append(clarification)
    context += older_messages + recent_messages
    logger.info(f"Returning context of length {len(context)}")
    return context
