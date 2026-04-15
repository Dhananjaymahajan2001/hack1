from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Streamlit Cloud/runtime fix for protobuf descriptor errors through Chroma/OpenTelemetry import chain.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import chromadb
import numpy as np
import tiktoken
from rapidfuzz import process
from sentence_transformers import SentenceTransformer

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import AzureChatOpenAI
except Exception:  # pragma: no cover
    AzureChatOpenAI = None
    HumanMessage = None
    SystemMessage = None


class RetrievalTools:
    """
    ChromaDB-based retrieval agent with query expansion, metadata filtering, and RRF.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_dir))
        self.collection = self.client.get_or_create_collection(name=self.settings.chroma_collection)
        self.embedding_model = self._init_embedding_model()
        self.use_local_embeddings = self.embedding_model is not None

        self.metadata_fields = self._infer_metadata_fields()
        self.metadata_cache = self._load_metadata()
        self.llm = self._init_optional_llm()

    def _init_embedding_model(self) -> SentenceTransformer | None:
        try:
            return SentenceTransformer(self.settings.embedding_model_name)
        except Exception as exc:
            logger.warning(
                "Embedding model init failed (%s). Falling back to Chroma text-query mode.",
                exc,
            )
            return None

    def _init_optional_llm(self):
        if AzureChatOpenAI is None:
            return None
        api_key = os.environ.get("GPT4o_API_KEY", "")
        endpoint = os.environ.get("GPT4o_DEPLOYMENT_ENDPOINT", "")
        if not api_key or not endpoint:
            return None
        try:
            return AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_key=api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=endpoint,
                temperature=0,
            )
        except Exception:
            return None

    def _infer_metadata_fields(self) -> list[str]:
        sample = self.collection.get(include=["metadatas"], limit=1)
        metadatas = sample.get("metadatas", []) if sample else []
        if metadatas and metadatas[0]:
            return list(metadatas[0].keys())
        # default fields for document assistant
        return ["doc_name", "section", "source", "timestamp"]

    def _load_metadata(self) -> dict[str, list[str]]:
        metadata: dict[str, list[str]] = {field: [] for field in self.metadata_fields}
        try:
            sample = self.collection.get(include=["metadatas"], limit=self.settings.metadata_sample_size)
            metadatas = sample.get("metadatas", []) if sample else []
            for item in metadatas:
                if not item:
                    continue
                for field in self.metadata_fields:
                    value = item.get(field)
                    if value not in ["-", None, ""]:
                        metadata[field].append(str(value))
            for field in metadata:
                metadata[field] = sorted(set(metadata[field]))
        except Exception as exc:
            logger.warning("Metadata cache load failed: %s", exc)
        return metadata

    def fetch_all_records(
        self,
        output_fields: list[str] | None = None,
        batch_size: int = 500,
        max_records: int = 5000,
    ) -> list[dict[str, Any]]:
        all_results: list[dict[str, Any]] = []
        offset = 0

        while offset < max_records:
            limit = min(batch_size, max_records - offset)
            include_fields = ["documents", "metadatas"]
            if output_fields and "embeddings" in output_fields:
                include_fields.append("embeddings")
            batch = self.collection.get(include=include_fields, limit=limit, offset=offset)
            ids = batch.get("ids", [])
            docs = batch.get("documents", [])
            metas = batch.get("metadatas", [])
            if not ids:
                break

            for idx, doc_id in enumerate(ids):
                all_results.append(
                    {
                        "id": doc_id,
                        "document": docs[idx] if idx < len(docs) else "",
                        "metadata": metas[idx] if idx < len(metas) else {},
                    }
                )

            offset += len(ids)
            logger.info("Retrieved %s / %s records", len(all_results), max_records)

        return all_results

    def get_embedding(self, data: list[str], batch_size: int = 8) -> list[list[float]]:
        if self.embedding_model is None:
            raise RuntimeError("Local embedding model unavailable.")
        emb = self.embedding_model.encode(
            data,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb_list = emb.tolist() if isinstance(emb, np.ndarray) else list(emb)
        return emb_list

    def _resolve_typos(self, value: str, field: str) -> str:
        valid_values = self.metadata_cache.get(field, [])
        if not valid_values:
            return value
        match = process.extractOne(value, valid_values)
        if not match:
            return value
        candidate, score, _ = match
        return candidate if score > 75 else value

    def extract_json_from_prompt(self, response_text: str) -> list[dict[str, Any]] | None:
        try:
            array_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if array_match:
                parsed = json.loads(array_match.group(0))
                if isinstance(parsed, list):
                    return parsed

            obj_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if obj_match:
                parsed = json.loads(obj_match.group(0))
                if isinstance(parsed, dict):
                    return [parsed]
            return None
        except Exception:
            return None

    def _call_llm_for_filters(self, user_queries: list[str]) -> list[dict[str, Any]] | None:
        if not self.llm or not SystemMessage or not HumanMessage:
            return None

        limited_metadata = {field: values[:20] for field, values in self.metadata_cache.items()}
        messages = [
            SystemMessage(
                content=(
                    "Convert user search queries into a JSON array. Each item must have keys: "
                    '"query_text" and "filters". Filters keys allowed: '
                    "doc_name, section, source, timestamp_start, timestamp_end. "
                    "Output JSON only."
                )
            ),
            HumanMessage(
                content=(
                    f"Queries: {json.dumps(user_queries)}\n"
                    f"Available values (sample): {json.dumps(limited_metadata)}\n"
                    "Output JSON array only."
                )
            ),
        ]
        try:
            answer = self.llm.invoke(messages)
            output = answer.content.strip()
            if output.startswith("```"):
                output = re.sub(r"^```(?:json)?\n?", "", output)
                output = re.sub(r"\n?```$", "", output).strip()
            return self.extract_json_from_prompt(output)
        except Exception as exc:
            logger.warning("LLM filter parse failed: %s", exc)
            return None

    def _build_where_filter(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []

        for field in self.metadata_fields:
            value = filters.get(field)
            if value:
                corrected = self._resolve_typos(str(value), field)
                clauses.append({field: {"$eq": corrected}})

        ts_start = filters.get("timestamp_start")
        ts_end = filters.get("timestamp_end")
        if ts_start:
            clauses.append({"timestamp": {"$gte": str(ts_start)}})
        if ts_end:
            clauses.append({"timestamp": {"$lte": str(ts_end)}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _simple_filter_parse(self, query: str) -> dict[str, str]:
        filters: dict[str, str] = {}
        mappings = {"doc": "doc_name", "section": "section", "source": "source"}
        lowered = query.lower()
        for token, field in mappings.items():
            marker = f"{token}:"
            if marker in lowered:
                try:
                    value = lowered.split(marker, 1)[1].strip().split(" ", 1)[0]
                    if value:
                        filters[field] = value
                except Exception:
                    continue
        return filters

    def get_filters(self, user_queries: list[str]) -> list[dict[str, Any] | None]:
        parsed = self._call_llm_for_filters(user_queries)
        if not parsed:
            return [self._build_where_filter(self._simple_filter_parse(q)) for q in user_queries]

        where_list: list[dict[str, Any] | None] = []
        for i, query in enumerate(user_queries):
            item = parsed[i] if i < len(parsed) else {"filters": {}}
            filters = item.get("filters", {}) if isinstance(item, dict) else {}
            where_list.append(self._build_where_filter(filters))
            logger.info("Query filter for '%s': %s", query, where_list[-1])

        while len(where_list) < len(user_queries):
            where_list.append(None)
        return where_list[: len(user_queries)]

    def _generate_query_variants(self, original_query: str) -> list[str]:
        if self.llm and SystemMessage and HumanMessage:
            try:
                messages = [
                    SystemMessage(
                        content=(
                            "Generate exactly 4 concise search query variants. "
                            "Return plain text, one query per line, without numbering."
                        )
                    ),
                    HumanMessage(content=f"Original query: {original_query}"),
                ]
                resp = self.llm.invoke(messages)
                variants = [line.strip(" -\t") for line in resp.content.splitlines() if line.strip()]
                if variants:
                    return variants[:4]
            except Exception as exc:
                logger.warning("LLM query expansion failed: %s", exc)

        base = original_query.strip()
        return [
            base,
            f"{base} official circular details",
            f"{base} policy summary",
            f"{base} eligibility and process",
        ]

    def search_chromadb(
        self,
        vector_or_query: list[float] | str,
        top_k: int,
        where_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if self.use_local_embeddings:
            result = self.collection.query(
                query_embeddings=[vector_or_query],  # type: ignore[list-item]
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        else:
            result = self.collection.query(
                query_texts=[str(vector_or_query)],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        formatted: list[dict[str, Any]] = []
        for i, doc_id in enumerate(ids):
            formatted.append(
                {
                    "id": doc_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": float(distances[i]) if i < len(distances) else 0.0,
                }
            )
        return formatted

    def are_results_similar(self, all_results: list[list[dict[str, Any]]], threshold: float = 0.7) -> bool:
        seen_ids: set[str] = set()
        total = 0
        for result in all_results:
            ids = {str(doc["id"]) for doc in result}
            seen_ids.update(ids)
            total += len(ids)
        if total == 0:
            return False
        unique_ratio = len(seen_ids) / total
        return unique_ratio < (1 - threshold)

    def reciprocal_rank_fusion(self, results: list[list[dict[str, Any]]], k: int = 100) -> list[tuple[dict[str, Any], float]]:
        fused_scores: dict[str, dict[str, Any]] = {}
        for docs in results:
            for rank, doc in enumerate(docs):
                doc_id = str(doc["id"])
                score = 1 / (rank + k)
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = {"doc": doc, "score": 0.0}
                fused_scores[doc_id]["score"] += score

        return sorted(
            [(v["doc"], v["score"]) for v in fused_scores.values()],
            key=lambda x: x[1],
            reverse=True,
        )

    def build_llm_context_from_reranked(
        self,
        reranked: list[tuple[dict[str, Any], float]],
        max_tokens: int = 4000,
        model: str = "gpt-4o",
    ) -> str:
        context_lines: list[str] = []
        total_tokens = 0
        encoder = tiktoken.encoding_for_model(model)

        for doc, score in reranked:
            meta = doc.get("metadata", {})
            line = (
                f"Doc: {meta.get('doc_name', 'UnknownDoc')} | "
                f"Section: {meta.get('section', 'N/A')} | "
                f"Source: {meta.get('source', 'N/A')} | "
                f"RRF: {score:.4f}\n"
                f"Content: {doc.get('content', '')}\n---\n"
            )
            block_tokens = len(encoder.encode(line))
            if total_tokens + block_tokens > max_tokens:
                break
            context_lines.append(line)
            total_tokens += block_tokens

        return "".join(context_lines)

    def search_chunks(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_variants = self._generate_query_variants(query)
        where_filters = self.get_filters(query_variants)
        vectors_or_queries: list[list[float] | str]
        if self.use_local_embeddings:
            vectors_or_queries = self.get_embedding(query_variants, batch_size=4)
        else:
            vectors_or_queries = query_variants

        with ThreadPoolExecutor(max_workers=min(4, len(vectors_or_queries))) as executor:
            all_results = list(
                executor.map(
                    lambda args: self.search_chromadb(*args),
                    zip(vectors_or_queries, [max(30, k * 4)] * len(vectors_or_queries), where_filters),
                )
            )

        if self.are_results_similar(all_results):
            reranked = [(doc, 1.0) for doc in all_results[0]]
        else:
            reranked = self.reciprocal_rank_fusion(all_results)

        return [doc for doc, _ in reranked[:k]]

    def query(self, user_query: str, top_k: int = 6) -> str:
        docs = self.search_chunks(user_query, k=top_k)
        reranked = [(d, 1.0) for d in docs]
        return self.build_llm_context_from_reranked(reranked)

    def upsert_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not documents:
            return
        if self.use_local_embeddings:
            embeddings = self.get_embedding(documents, batch_size=16)
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        self.metadata_fields = self._infer_metadata_fields()
        self.metadata_cache = self._load_metadata()
