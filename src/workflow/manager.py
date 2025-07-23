# Placeholder for WorkflowManager
import logging
import json
from datetime import datetime
import re
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from .state import AgentState, find_last_message_by_name, find_last_human_message
from .database import get_schema_description, execute_query, check_sql_syntax
from .sql_processor import generate_sql_from_nl, attempt_fix_sql
from .utils.entity_resolution import find_ambiguous_entities, resolve_entities
from .llm_factory import get_llm
from .utils.memory_utils import _prepare_llm_context
from . import config as config_module

logger = logging.getLogger(__name__)

FORMAT_ANSWER_BASE_PROMPT = """
Jij bent Fred, een hartstochtelijke fan van de voetbalclub Feyenoord uit Rotterdam-Zuid.
Jouw doel is om vragen over Feyenoord te beantwoorden.
Neem de vraag en de resultaten uit de database om een kort en bondig antwoord op de vraag te formuleren.
Houd rekening met de volgende richtlijnen:
- Ga nooit antwoorden verzinnen op de vraag van de gebruiker. Gebruik uitsluitend de informatie die jou gegeven hebt.
- Geef altijd een antwoord in de eerste persoon, alsof jij Fred bent.
- Als de query geen resultaten opleverde, wees daar dan eerlijk over en zeg dan dat je het antwoord niet weet op de vraag.
- Refereer nooit naar technische details zoals SQL-query's of database-informatie.
"""

FORMAT_ERROR_PROMPT = FORMAT_ANSWER_BASE_PROMPT + """
- Als een foutmelding is opgetreden, laat dan de technische details achterwege.
- Als je geen antwoord kunt geven, gebruik dan de informatie die jou gegeven is om verduidelijking te vragen. Gebruik hiervoor het database-schema.
- Als je geen verduidelijke vragen kunt stellen, geef dan een lijst (in Markdown-formaat) met suggesties van maximaal 3 vragen die je wel kunt beantwoorden. Gebruik hiervoor het database-schema.
"""

class WorkflowManager:
    def __init__(self, format_output: bool = True):
        self.format_output = format_output
        
        self.max_sql_fix_attempts = getattr(config_module, "MAX_SQL_FIX_ATTEMPTS", 1)
        self.llm = get_llm()
        self.graph = self._compile_graph()

    def get_graph(self):
        return self.graph

    def _canonicalize_query(self, user_query: str, resolved_entities: dict) -> str:
        result = ' '.join(user_query.split())
        for mention, canonical in resolved_entities.items():
            clean_mention = ' '.join(mention.split())
            pattern = r'\b' + re.escape(clean_mention) + r'\b'
            result = re.sub(pattern, canonical, result, flags=re.IGNORECASE)
        logger.info(f"Canonicalized query: '{result}'")
        return result

    async def get_schema_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Getting schema")
        if not state.get("schema"):
            try:
                db_url = getattr(config_module, "FEYOD_DATABASE_URL", None)
                logger.info(f"DATABASE URL: '{db_url}'")
                schema = await get_schema_description(db_url, self.llm)
                if not schema or "Error" in schema:
                    raise ValueError(f"Failed to retrieve schema: {schema}")
                return {
                    "messages": [AIMessage(content="Schema loaded.", name="schema")],
                    "schema": schema,
                    "schema_timestamp": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.exception("Error in get_schema_node")
                return {"messages": [AIMessage(content=f"Fatal error getting schema: {e}", name="error")]}
        return {}

    async def clarify_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Clarifying ambiguous entities")
        nl_query_msg = find_last_human_message(state["messages"])
        if not nl_query_msg:
            return {"messages": [AIMessage(content="Could not find user query for clarification.", name="error")]}

        ambiguous_entities = await find_ambiguous_entities(nl_query_msg.content)
        new_resolved = await resolve_entities(nl_query_msg.content)
        
        resolved_entities = state.get("resolved_entities", {})
        resolved_entities.update(new_resolved)
        
        if ambiguous_entities:
            clarification_text = f"I found multiple possibilities for: {', '.join(ambiguous_entities)}. Could you be more specific?"
            return { "messages": [AIMessage(content=clarification_text, name="clarify")], "resolved_entities": resolved_entities }
        return { "messages": [AIMessage(content="Entities clarified.", name="clarified")], "resolved_entities": resolved_entities }

    async def generate_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Generating SQL query")
        context_messages = await _prepare_llm_context(state["messages"])
        nl_query_msg = find_last_human_message(context_messages)
        
        if not nl_query_msg or not state.get("schema"):
            return {"messages": [AIMessage(content="Missing context for SQL generation.", name="error")]}

        resolved_entities = state.get("resolved_entities", {})
        canonical_query = self._canonicalize_query(nl_query_msg.content, resolved_entities)

        try:
            sql = await generate_sql_from_nl(canonical_query, state["schema"], context_messages)
            return {"messages": [AIMessage(content=sql, name="sql_query")]}
        except Exception as e:
            logger.exception("Error generating SQL")
            return {"messages": [AIMessage(content=f"Error generating SQL: {e}", name="error")]}

    async def check_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Checking SQL query")
        sql_msg = find_last_message_by_name(state["messages"], "sql_query")
        if not sql_msg:
            return {"messages": [AIMessage(content="No SQL query found to check.", name="error")]}
        
        try:
            db_url = getattr(config_module, "FEYOD_DATABASE_URL", None)
            is_valid, error = await check_sql_syntax(sql_msg.content, db_url)
            if is_valid:
                return {"messages": [AIMessage(content="Syntax check OK", name="check_result")]}
            return {"messages": [AIMessage(content=error or "Invalid SQL syntax", name="error")]}
        except Exception as e:
            logger.exception("Error checking SQL syntax")
            return {"messages": [AIMessage(content=f"Error checking SQL: {e}", name="error")]}

    async def execute_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Executing SQL query")
        sql_msg = find_last_message_by_name(state["messages"], "sql_query")
        if not sql_msg:
            return {"messages": [AIMessage(content="No SQL query found to execute.", name="error")]}
            
        try:
            db_url = getattr(config_module, "FEYOD_DATABASE_URL", None)
            results = await execute_query(sql_msg.content, db_url)
            return {"messages": [AIMessage(content=json.dumps(results), name="results")]}
        except Exception as e:
            logger.exception("Error executing query")
            return {"messages": [AIMessage(content=f"Error executing query: {e}", name="error")]}

    async def fix_query_node(self, state: AgentState) -> Dict[str, Any]:
        logger.warning("Node: Attempting to fix SQL query")
        context_messages = await _prepare_llm_context(state["messages"])
        invalid_sql_msg = find_last_message_by_name(context_messages, "sql_query")
        error_msg = find_last_message_by_name(context_messages, "error")
        original_nl_query_msg = find_last_human_message(context_messages)

        if not all([invalid_sql_msg, error_msg, state.get("schema"), original_nl_query_msg]):
            return {"messages": [AIMessage(content="Cannot fix query: Missing context.", name="error")]}
        
        fix_attempts = state.get("fix_attempts", 0) + 1
        try:
            fixed_sql = await attempt_fix_sql(
                invalid_sql=invalid_sql_msg.content,
                error_message=error_msg.content,
                schema=state["schema"],
                original_nl_query=original_nl_query_msg.content
            )
            return {"messages": [AIMessage(content=fixed_sql, name="sql_query")], "fix_attempts": fix_attempts}
        except Exception as e:
            logger.exception("Error attempting to fix SQL")
            return {"messages": [AIMessage(content=f"Failed to attempt fix: {e}", name="error")], "fix_attempts": fix_attempts}

    async def format_answer_node(self, state: AgentState) -> Dict[str, Any]:
        logger.info("Node: Formatting final answer")
        question_msg = find_last_human_message(state["messages"])
        results_msg = find_last_message_by_name(state["messages"], "results")
        error_msg = find_last_message_by_name(state["messages"], "error")

        if not self.llm or not question_msg:
            return {"messages": [AIMessage(content="Error: Cannot format answer due to missing LLM or original question.")]}

        final_answer_content = "Sorry, an unexpected error occurred."
        if results_msg:
            try:
                results = json.loads(results_msg.content)
                prompt_template = ChatPromptTemplate.from_messages([("system", FORMAT_ANSWER_BASE_PROMPT), ("human", "Vraag: {question}\n\nResultaten:\n{results}")])
                if not results:
                    prompt = prompt_template.format(question=question_msg.content, results="Geen resultaten gevonden.")
                else:
                    formatted_results = "\n".join([str(row) for row in results])
                    prompt = prompt_template.format(question=question_msg.content, results=formatted_results)
                
                ai_response = await self.llm.ainvoke(prompt)
                final_answer_content = ai_response.content
            except Exception as e:
                logger.exception("Error formatting answer with results")
                final_answer_content = "Sorry, ik kon de resultaten niet verwerken."

        elif error_msg:
            try:
                prompt_template = ChatPromptTemplate.from_messages([("system", FORMAT_ERROR_PROMPT), ("human", "Vraag: {question}\n\nFoutmelding: {error}\n\nDatabase-schema:\n{schema}")])
                prompt = prompt_template.format(question=question_msg.content, error=error_msg.content, schema=state.get("schema"))
                ai_response = await self.llm.ainvoke(prompt)
                final_answer_content = ai_response.content
            except Exception as e:
                logger.exception("Error formatting answer with error")
                final_answer_content = "Sorry, er is een fout opgetreden bij het verwerken van de fout."

        return {"messages": [AIMessage(content=final_answer_content)]}

    def should_fix_or_execute(self, state: AgentState) -> str:
        if find_last_message_by_name(state["messages"], "check_result"):
            return "execute_query"
        if state.get("fix_attempts", 0) >= self.max_sql_fix_attempts:
            logger.error(f"Exceeded max fix attempts ({self.max_sql_fix_attempts}).")
            return "error_handler"
        return "fix_query"

    def after_clarification(self, state: AgentState) -> str:
        if find_last_message_by_name(state["messages"], "clarify"):
            return END
        return "generate_query"
        
    def after_execution(self, state: AgentState) -> str:
        if self.format_output:
            return "format_answer"
        return END

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        nodes = {
            "get_schema": self.get_schema_node, "clarify": self.clarify_node,
            "generate_query": self.generate_query_node, "check_query": self.check_query_node,
            "execute_query": self.execute_query_node, "fix_query": self.fix_query_node,
            "format_answer": self.format_answer_node,
            "error_handler": lambda state: {"messages": [AIMessage(content="I'm sorry, but I was unable to process your request.")]}
        }
        for name, node in nodes.items():
            workflow.add_node(name, node)

        workflow.set_entry_point("get_schema")
        workflow.add_edge("get_schema", "clarify")
        workflow.add_conditional_edges("clarify", self.after_clarification)
        workflow.add_edge("generate_query", "check_query")
        workflow.add_conditional_edges("check_query", self.should_fix_or_execute)
        workflow.add_edge("fix_query", "check_query")
        workflow.add_conditional_edges("execute_query", self.after_execution)
        workflow.add_edge("format_answer", END)
        workflow.add_edge("error_handler", END)
        return workflow

    def _compile_graph(self):
        logger.info("Compiling the workflow graph.")
        return self._create_workflow().compile() 