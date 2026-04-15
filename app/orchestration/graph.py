from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config.settings import get_settings
from app.mcp_tools.retrieval_tools import RetrievalTools
from app.services.sarvam_client import SarvamClient


class AssistantState(TypedDict, total=False):
    user_query: str
    forced_lang: str
    feedback_context: str
    conversation_context: list[dict]
    input_lang: str
    pivot_query: str
    retrieved_chunks: list[dict]
    answer_pivot: str
    final_answer: str
    citations: list[str]
    error: str


class Orchestrator:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.retrieval = RetrievalTools()
        self.sarvam = SarvamClient()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AssistantState)

        workflow.add_node("detect_language", self.detect_language)
        workflow.add_node("translate_to_pivot", self.translate_to_pivot)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("translate_back", self.translate_back)

        workflow.set_entry_point("detect_language")
        workflow.add_edge("detect_language", "translate_to_pivot")
        workflow.add_edge("translate_to_pivot", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "translate_back")
        workflow.add_edge("translate_back", END)

        return workflow.compile()

    async def run(
        self,
        query: str,
        forced_lang: str | None = None,
        feedback_context: str = "",
        conversation_context: list[dict] | None = None,
    ) -> AssistantState:
        initial_state: AssistantState = {"user_query": query}
        if forced_lang:
            initial_state["forced_lang"] = forced_lang
        if feedback_context:
            initial_state["feedback_context"] = feedback_context
        if conversation_context:
            initial_state["conversation_context"] = conversation_context
        return await self.graph.ainvoke(initial_state)

    async def detect_language(self, state: AssistantState) -> AssistantState:
        user_query = state["user_query"]
        forced = state.get("forced_lang")
        if forced and forced != "auto":
            input_lang = forced
        else:
            input_lang = await self.sarvam.detect_language(user_query)
        state["input_lang"] = input_lang
        return state

    async def translate_to_pivot(self, state: AssistantState) -> AssistantState:
        query = state["user_query"]
        input_lang = state.get("input_lang", "en-IN")
        pivot_lang = "en-IN"
        state["pivot_query"] = await self.sarvam.translate_text(query, input_lang, pivot_lang)
        return state

    async def retrieve_context(self, state: AssistantState) -> AssistantState:
        pivot_query = state["pivot_query"]
        chunks = self.retrieval.search_chunks(pivot_query, k=5)
        state["retrieved_chunks"] = chunks
        citations = []
        for item in chunks:
            meta = item.get("metadata", {})
            doc_name = meta.get("doc_name", "UnknownDoc")
            section = meta.get("section", "N/A")
            citations.append(f"{doc_name} | section: {section}")
        state["citations"] = citations
        return state

    async def generate_answer(self, state: AssistantState) -> AssistantState:
        pivot_query = state["pivot_query"]
        chunks = state.get("retrieved_chunks", [])
        chunk_texts = [c.get("content", "") for c in chunks]
        feedback_context = state.get("feedback_context", "")
        conversation_context = state.get("conversation_context", [])
        answer = await self.sarvam.chat_completion(
            pivot_query,
            chunk_texts,
            feedback_context=feedback_context,
            conversation_context=conversation_context,
        )
        state["answer_pivot"] = answer
        return state

    async def translate_back(self, state: AssistantState) -> AssistantState:
        input_lang = state.get("input_lang", self.settings.default_language)
        answer_pivot = state.get("answer_pivot", "")
        final_answer = await self.sarvam.translate_text(answer_pivot, "en-IN", input_lang)
        state["final_answer"] = final_answer
        return state
