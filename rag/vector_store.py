"""Vector store utilities for retention knowledge retrieval (RAG)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.llm import get_embedding_model


# Compatibility fallback for certain protobuf/chromadb runtime combinations.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


RAG_DIR = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_PATH = RAG_DIR / "knowledge.txt"
DEFAULT_PERSIST_DIR = RAG_DIR / "chroma_db"
COLLECTION_NAME = "player_churn_knowledge"


def _load_knowledge_text(knowledge_path: Path) -> str:
	"""Read knowledge file and return its content."""

	if not knowledge_path.exists():
		raise FileNotFoundError(f"Knowledge file not found at: {knowledge_path}")

	text = knowledge_path.read_text(encoding="utf-8").strip()
	if not text:
		raise ValueError("Knowledge file is empty. Add retention strategies first.")
	return text


def _build_documents(text: str) -> List[Document]:
	"""Split raw knowledge text into chunked documents for embedding/indexing."""

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=450,
		chunk_overlap=80,
		separators=["\n\n", "\n", ". ", " ", ""],
	)
	chunks = [chunk for chunk in splitter.split_text(text) if chunk.strip()]

	return [
		Document(page_content=chunk, metadata={"source": "knowledge.txt", "chunk_id": idx})
		for idx, chunk in enumerate(chunks)
	]


def build_vector_store(
	knowledge_path: Path | None = None,
	persist_dir: Path | None = None,
	force_rebuild: bool = False,
) -> Any:
	"""Create or load a persisted Chroma vector store.

	Parameters
	----------
	knowledge_path:
		Path to text knowledge base.
	persist_dir:
		Path where Chroma persists vectors.
	force_rebuild:
		If True, clears old collection content and reindexes from scratch.
	"""

	knowledge_path = knowledge_path or DEFAULT_KNOWLEDGE_PATH
	persist_dir = persist_dir or DEFAULT_PERSIST_DIR
	persist_dir.mkdir(parents=True, exist_ok=True)

	# Lazy import so the base prediction app remains usable even if optional
	# RAG runtime dependencies are unavailable at import time.
	from langchain_chroma import Chroma

	vector_store = Chroma(
		collection_name=COLLECTION_NAME,
		embedding_function=get_embedding_model(),
		persist_directory=str(persist_dir),
	)

	current_count = vector_store._collection.count()
	if force_rebuild and current_count > 0:
		vector_store.delete(ids=vector_store.get(include=[]).get("ids", []))
		current_count = 0

	if current_count == 0:
		raw_text = _load_knowledge_text(knowledge_path)
		documents = _build_documents(raw_text)
		vector_store.add_documents(documents)

	return vector_store


def retrieve_relevant_strategies(query: str, k: int = 4) -> List[str]:
	"""Retrieve top-k strategy snippets relevant to the given query."""

	query = (query or "").strip()
	if not query:
		return []

	try:
		vector_store = build_vector_store()
		matches = vector_store.similarity_search(query, k=k)
		return [doc.page_content for doc in matches]
	except Exception:
		# Fallback retrieval when provider embeddings are unavailable.
		# Uses lightweight keyword overlap scoring over chunked knowledge.
		raw_text = _load_knowledge_text(DEFAULT_KNOWLEDGE_PATH)
		documents = _build_documents(raw_text)

		query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
		scored_docs = []

		for doc in documents:
			content = doc.page_content
			content_terms = set(re.findall(r"[a-zA-Z0-9]+", content.lower()))
			overlap_score = len(query_terms & content_terms)
			scored_docs.append((overlap_score, content))

		scored_docs.sort(key=lambda item: item[0], reverse=True)
		top_docs = [content for score, content in scored_docs if score > 0][:k]

		if not top_docs:
			top_docs = [doc.page_content for doc in documents[:k]]

		return top_docs
