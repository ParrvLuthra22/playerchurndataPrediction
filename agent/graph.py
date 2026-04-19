"""LangGraph workflow for AI-powered churn explanation and retention advice."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.vector_store import retrieve_relevant_strategies
from utils.llm import get_chat_model


class AgentState(TypedDict, total=False):
    """Shared graph state passed between all nodes."""

    player_data: Dict[str, Any]
    prediction_label: str
    analysis: str
    retrieval_query: str
    retrieved_docs: List[str]
    reason: str
    recommendations: List[str]
    final_response: str


def analyze_node(state: AgentState) -> AgentState:
    """Analyze prediction and prepare a retrieval query."""

    player_data = state.get("player_data", {})
    prediction_label = state.get("prediction_label", "Unknown Risk")

    compact_features = ", ".join(
        f"{key}={value}" for key, value in list(player_data.items())[:8]
    )
    analysis = (
        f"Model prediction is '{prediction_label}'. "
        f"Player profile snapshot: {compact_features}. "
        "Need retention strategies tuned to this risk pattern."
    )

    retrieval_query = (
        f"{prediction_label} churn player retention strategy based on profile: "
        f"{compact_features}"
    )

    return {
        "analysis": analysis,
        "retrieval_query": retrieval_query,
    }


def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant retention knowledge from the vector store."""

    query = state.get("retrieval_query", state.get("analysis", ""))
    retrieved_docs = retrieve_relevant_strategies(query=query, k=4)
    return {"retrieved_docs": retrieved_docs}


def _parse_agent_response(response_text: str) -> tuple[str, List[str]]:
    """Extract reason and recommendation bullets from model output."""

    reason = "Unable to generate a detailed reason from model output."
    recommendations: List[str] = []

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    in_recommendation_section = False

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("reason:"):
            reason = line.split(":", 1)[1].strip() or reason
            in_recommendation_section = False
            continue

        if lowered.startswith(("recommendation:", "recommendations:")):
            in_recommendation_section = True
            trailing = line.split(":", 1)[1].strip()
            if trailing:
                recommendations.append(trailing.lstrip("-•*0123456789. ").strip())
            continue

        if in_recommendation_section and line.startswith(("-", "•", "*")):
            recommendations.append(line.lstrip("-•*0123456789. ").strip())

    if not recommendations:
        recommendations = [
            "Offer a targeted re-engagement reward.",
            "Trigger personalized notifications based on play history.",
            "Schedule a follow-up campaign if activity remains low.",
        ]

    return reason, recommendations


def generate_node(state: AgentState) -> AgentState:
	"""Generate explanation and actionable recommendations using LLM."""

	chat_model = get_chat_model()

	player_data = state.get("player_data", {})
	prediction_label = state.get("prediction_label", "Unknown Risk")
	analysis = state.get("analysis", "")
	retrieved_docs = state.get("retrieved_docs", [])

	knowledge_context = "\n\n".join(
		f"Context {idx + 1}: {doc}" for idx, doc in enumerate(retrieved_docs)
	)

	system_prompt = (
		"You are a senior player retention strategist. "
		"Given player features, model prediction, and retrieved strategy snippets, "
		"generate a concise explanation and 3 practical recommendations. "
		"Return strictly in this format:\n"
		"Prediction: <label>\n\n"
		"Reason:\n"
		"<1 concise paragraph>\n\n"
		"Recommendation:\n"
		"- <action 1>\n"
		"- <action 2>\n"
		"- <action 3>"
	)

	user_prompt = (
		f"Prediction Label: {prediction_label}\n"
		f"Analysis: {analysis}\n"
		f"Player Data (JSON): {json.dumps(player_data, default=str)}\n\n"
		f"Retrieved Knowledge:\n{knowledge_context}"
	)

	llm_response = chat_model.invoke(
		[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]
	)

	response_text = (
		llm_response.content
		if isinstance(llm_response.content, str)
		else str(llm_response.content)
	)
	reason, recommendations = _parse_agent_response(response_text)

	return {
		"reason": reason,
		"recommendations": recommendations,
		"final_response": response_text,
	}


@lru_cache(maxsize=1)
def build_churn_advisor_graph():
	"""Create and compile the churn advisor graph."""

	workflow = StateGraph(AgentState)
	workflow.add_node("analyze", analyze_node)
	workflow.add_node("retrieve", retrieve_node)
	workflow.add_node("generate", generate_node)

	workflow.add_edge(START, "analyze")
	workflow.add_edge("analyze", "retrieve")
	workflow.add_edge("retrieve", "generate")
	workflow.add_edge("generate", END)

	return workflow.compile()


def run_churn_advisor(player_data: Dict[str, Any], prediction_label: str) -> Dict[str, Any]:
    """Run the LangGraph workflow for a single player row."""

    app = build_churn_advisor_graph()
    final_state = app.invoke(
        {
            "player_data": player_data,
            "prediction_label": prediction_label,
        }
    )

    return {
        "prediction": prediction_label,
        "reason": final_state.get("reason", "Reason unavailable."),
        "recommendations": final_state.get("recommendations", []),
        "raw_response": final_state.get("final_response", ""),
        "retrieved_docs": final_state.get("retrieved_docs", []),
    }
