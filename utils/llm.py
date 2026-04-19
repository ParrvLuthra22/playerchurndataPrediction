"""Utilities for Groq API chat and embedding clients.

This module centralizes model configuration so both the RAG layer and the
LangGraph agent can share the same settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class LLMSettings:
	"""Application-level LLM settings loaded from environment variables."""

	api_key: str
	base_url: str | None
	chat_model: str
	embedding_model: str
	temperature: float


def get_llm_settings() -> LLMSettings:
	"""Return LLM settings loaded from environment variables.

	Required env var:
	- GROQ_API_KEY

	Optional env vars:
	- GROQ_BASE_URL
	- GROQ_MODEL
	- GROQ_EMBEDDING_MODEL
	- GROQ_TEMPERATURE
	"""

	api_key = os.getenv("GROQ_API_KEY", "").strip()
	if not api_key:
		raise ValueError(
			"GROQ_API_KEY is not set. Add it to your environment before using "
			"the GenAI + RAG features."
		)

	return LLMSettings(
		api_key=api_key,
		base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip() or None,
		chat_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
		embedding_model=os.getenv("GROQ_EMBEDDING_MODEL", "text-embedding-3-small"),
		temperature=float(os.getenv("GROQ_TEMPERATURE", "0.25")),
	)


def get_chat_model() -> ChatOpenAI:
	"""Create a configured chat model client."""

	settings = get_llm_settings()
	return ChatOpenAI(
		model=settings.chat_model,
		temperature=settings.temperature,
		api_key=settings.api_key,
		base_url=settings.base_url,
	)


def get_embedding_model() -> OpenAIEmbeddings:
	"""Create a configured embedding model client."""

	settings = get_llm_settings()
	return OpenAIEmbeddings(
		model=settings.embedding_model,
		api_key=settings.api_key,
		base_url=settings.base_url,
		# Groq's OpenAI-compatible embeddings endpoint expects raw strings,
		# so disable token-length preprocessing that can send token arrays.
		check_embedding_ctx_length=False,
	)
