import pytest
import asyncio
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from unittest.mock import AsyncMock, patch
from common.utils.memory_utils import _prepare_llm_context
from common.config import CONTEXT_RECENT_MESSAGES_KEPT

def test_prepare_llm_context_few_messages(event_loop):
    messages = [HumanMessage(content=f"User {i}") for i in range(3)]
    context = event_loop.run_until_complete(_prepare_llm_context(messages))
    assert context == messages

def test_prepare_llm_context_below_threshold(event_loop):
    messages = [HumanMessage(content="A" * 1000) for _ in range(5)]
    context = event_loop.run_until_complete(_prepare_llm_context(messages))
    assert context == messages

@patch("common.utils.memory_utils.get_llm")
def test_prepare_llm_context_triggers_summarization(mock_get_llm, event_loop):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="SUMMARY")
    mock_get_llm.return_value = mock_llm
    from common.utils import memory_utils
    memory_utils.CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD = 1000
    memory_utils.CONTEXT_RECENT_MESSAGES_KEPT = 3
    messages = [HumanMessage(content="A" * 500) for _ in range(10)]
    context = event_loop.run_until_complete(_prepare_llm_context(messages))
    assert isinstance(context[0], SystemMessage)
    assert context[0].content == "SUMMARY"
    assert len(context) == memory_utils.CONTEXT_RECENT_MESSAGES_KEPT + 1
    for m in context[1:]:
        assert isinstance(m, HumanMessage)

@patch("common.utils.memory_utils.get_llm")
def test_prepare_llm_context_summarization_error(mock_get_llm, event_loop):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.side_effect = Exception("fail")
    mock_get_llm.return_value = mock_llm
    from common.utils import memory_utils
    memory_utils.CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD = 1000
    memory_utils.CONTEXT_RECENT_MESSAGES_KEPT = 3
    messages = [HumanMessage(content="A" * 500) for _ in range(10)]
    context = event_loop.run_until_complete(_prepare_llm_context(messages))
    assert all(isinstance(m, HumanMessage) for m in context)
    assert len(context) == memory_utils.CONTEXT_RECENT_MESSAGES_KEPT

@patch("common.utils.memory_utils.get_llm")
def test_prepare_llm_context_collapse_error_chains(mock_get_llm, event_loop):
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = AIMessage(content="SUMMARY")
    mock_get_llm.return_value = mock_llm
    from common.utils import memory_utils
    memory_utils.CONTEXT_OLDER_MESSAGES_CHAR_THRESHOLD = 1000
    memory_utils.CONTEXT_RECENT_MESSAGES_KEPT = 3
    # 8 errors, 2 meaningful (make errors long to exceed threshold)
    messages = [AIMessage(content=f"error{i} {'.'*500}", name="error") for i in range(8)] + [HumanMessage(content="User Q"), AIMessage(content="AI answer", name="sql_query")]
    context = event_loop.run_until_complete(_prepare_llm_context(messages))
    assert isinstance(context[0], SystemMessage)
    assert context[0].content == "SUMMARY"
    assert len(context) == memory_utils.CONTEXT_RECENT_MESSAGES_KEPT + 1
    error_count = sum(1 for m in context[1:] if getattr(m, "name", None) == "error")
    assert error_count == 1

def test_prepare_llm_context_loop_detection(event_loop):
    # All errors in recent buffer triggers clarification
    from common.utils.memory_utils import SystemMessage
    messages = [AIMessage(content=f"error{i}", name="error") for i in range(CONTEXT_RECENT_MESSAGES_KEPT)]
    context = event_loop.run_until_complete(_prepare_llm_context(messages))
    assert isinstance(context[0], SystemMessage)
    assert context[0].name == "clarification_request"
    assert "fouten" in context[0].content
