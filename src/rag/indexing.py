"""RAG ingestion and indexing helpers backed by LangChain and ChromaDB.

This module is intentionally not an agent. It only converts browser extraction
results into persistent vector-search chunks.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from src.agents.change_detection_agent import hash_text, normalize_url, objective_key
from src.tools.text_utils import clean_text


DEFAULT_COLLECTION_NAME = "research_rag"
DEFAULT_CHROMA_PATH = "data/chroma"
DEFAULT_CHUNK_SIZE = 1600
DEFAULT_CHUNK_OVERLAP = 240
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# "auto" prefers CUDA on RunPod, MPS on Apple Silicon, then CPU as fallback.
DEFAULT_EMBEDDING_DEVICE = "auto"


@dataclass(frozen=True)
class SourceRecord:
    url: str
    title: str
    task_ids: str
    query_contexts: str
    source_type: str
    source_quality: str
    content_hash: str
    content: str


class SentenceTransformerEmbeddingFunction:
    """Chroma-compatible embedding function backed by sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = clean_text(model_name or os.environ.get("RAG_EMBEDDING_MODEL")) or DEFAULT_EMBEDDING_MODEL
        # Set RAG_EMBEDDING_DEVICE=cuda in RunPod to force GPU embeddings.
        self.device = clean_text(os.environ.get("RAG_EMBEDDING_DEVICE")) or DEFAULT_EMBEDDING_DEVICE
        self._model = None

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = self.model()
        embeddings = model.encode(
            input,
            batch_size=max(1, to_int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE"), 32)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def name(self) -> str:
        """Stable Chroma embedding function name used when reopening collections."""
        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
        return f"sentence-transformers_{safe_model_name}"

    def model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` or `pip install -r requirements.txt`."
            ) from error

        self.device = select_embedding_device(self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


def select_embedding_device(requested_device: str = DEFAULT_EMBEDDING_DEVICE) -> str:
    """Return the embedding device, preferring GPU when requested_device is auto."""
    requested_device = clean_text(requested_device).lower()
    if requested_device and requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def index_research_results(
    browser_results: list[dict[str, Any]],
    research_plan: Optional[dict[str, Any]] = None,
    change_detection: Optional[dict[str, Any]] = None,
    chroma_path: Union[str, Path] = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict[str, Any]:
    """Index added/changed browser sources into ChromaDB.

    URLs absent from the current run are preserved in ChromaDB because a later
    research run may discover a different source set while older chunks remain
    useful.
    """

    collection = get_collection(chroma_path, collection_name)
    change_detection = change_detection if isinstance(change_detection, dict) else {}
    research_plan = research_plan if isinstance(research_plan, dict) else {}
    objective = clean_text(change_detection.get("objective") or research_plan.get("objective"))
    history_key = clean_text(change_detection.get("history_key")) or objective_key(objective, research_plan)

    records = list(source_records(browser_results))
    added_urls = summary_urls(change_detection.get("added_sources"))
    changed_urls = summary_urls(change_detection.get("changed_sources"))
    removed_urls = summary_urls(change_detection.get("removed_sources"))
    index_all = bool(change_detection.get("first_run")) or collection_is_empty(collection)
    indexed_source_count = 0
    indexed_chunk_count = 0
    deleted_changed_source_count = 0
    skipped_source_count = 0

    for record in records:
        change_status = source_change_status(record.url, added_urls, changed_urls)
        if not index_all and change_status == "unchanged":
            skipped_source_count += 1
            continue

        existing_content_hash = source_content_hash_in_collection(collection, history_key, record.url)
        if existing_content_hash == record.content_hash:
            skipped_source_count += 1
            continue

        source_document = build_langchain_document(record, objective, history_key, change_status if not index_all else "indexed")
        chunk_documents = split_document(source_document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunk_documents:
            skipped_source_count += 1
            continue

        if existing_content_hash and existing_content_hash != record.content_hash:
            delete_source_chunks(collection, history_key, record.url)
            deleted_changed_source_count += 1

        ids = []
        documents = []
        metadatas = []
        for chunk_index, chunk_document_item in enumerate(chunk_documents):
            ids.append(chunk_id(history_key, record.url, chunk_index))
            documents.append(chunk_document_item.page_content)
            metadatas.append(
                clean_metadata(
                    {
                        **chunk_document_item.metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunk_documents),
                    }
                )
            )

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        indexed_source_count += 1
        indexed_chunk_count += len(chunk_documents)

    return {
        "status": "success",
        "chroma_path": str(chroma_path),
        "collection_name": collection_name,
        "history_key": history_key,
        "indexed_sources": indexed_source_count,
        "indexed_chunks": indexed_chunk_count,
        "deleted_changed_sources": deleted_changed_source_count,
        "preserved_removed_sources": len(removed_urls),
        "skipped_sources": skipped_source_count,
        "total_sources_seen": len(records),
    }


def get_collection(chroma_path: Union[str, Path], collection_name: str):
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            "chromadb is not installed. Install it with `pip install chromadb`."
        ) from error

    path = Path(chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=SentenceTransformerEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def build_langchain_document(record: SourceRecord, objective: str, history_key: str, change_status: str) -> Any:
    Document, _ = langchain_ingestion_classes()
    return Document(
        page_content=chunk_document(record, record.content),
        metadata=clean_metadata(
            {
                "history_key": history_key,
                "objective": objective,
                "url": record.url,
                "title": record.title,
                "task_ids": record.task_ids,
                "query_contexts": record.query_contexts,
                "source_type": record.source_type,
                "source_quality": record.source_quality,
                "content_hash": record.content_hash,
                "change_status": change_status,
            }
        ),
    )


def split_document(document: Any, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Any]:
    _, RecursiveCharacterTextSplitter = langchain_ingestion_classes()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(400, chunk_size),
        chunk_overlap=max(0, min(chunk_overlap, chunk_size // 2)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents([document])


def langchain_ingestion_classes() -> tuple[Any, Any]:
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as error:
        raise RuntimeError(
            "LangChain ingestion dependencies are not installed. Install them with "
            "`pip install langchain-core langchain-text-splitters` or `pip install -r requirements.txt`."
        ) from error
    return Document, RecursiveCharacterTextSplitter


def source_records(browser_results: list[dict[str, Any]]) -> Iterable[SourceRecord]:
    grouped: dict[str, dict[str, Any]] = {}
    for result in browser_results:
        task_id = clean_text(result.get("task_id"))
        query_context = clean_text(result.get("query_context"))
        for source in result.get("sources", []):
            if not isinstance(source, dict):
                continue
            url = normalize_url(clean_text(source.get("url")))
            content = clean_text(source.get("full_content") or source.get("content_preview"))
            if not url or not content:
                continue

            record = grouped.setdefault(
                url,
                {
                    "url": url,
                    "title": clean_text(source.get("title")),
                    "task_ids": [],
                    "query_contexts": [],
                    "source_type": clean_text(source.get("source_type")),
                    "source_quality": clean_text(source.get("source_quality")),
                    "content": content,
                },
            )
            if task_id and task_id not in record["task_ids"]:
                record["task_ids"].append(task_id)
            if query_context and query_context not in record["query_contexts"]:
                record["query_contexts"].append(query_context)

            if len(content) > len(record["content"]):
                record["content"] = content
                record["title"] = clean_text(source.get("title")) or record["title"]
                record["source_type"] = clean_text(source.get("source_type")) or record["source_type"]
                record["source_quality"] = clean_text(source.get("source_quality")) or record["source_quality"]

    for record in grouped.values():
        content = record["content"]
        yield SourceRecord(
            url=record["url"],
            title=record["title"],
            task_ids=", ".join(record["task_ids"]),
            query_contexts=" | ".join(record["query_contexts"]),
            source_type=record["source_type"],
            source_quality=record["source_quality"],
            content_hash=hash_text(content),
            content=content,
        )


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    Document, _ = langchain_ingestion_classes()
    document = Document(page_content=clean_text(text), metadata={})
    return [chunk.page_content for chunk in split_document(document, chunk_size=chunk_size, chunk_overlap=overlap)]


def chunk_document(record: SourceRecord, chunk: str) -> str:
    title = record.title or record.url
    return f"Source: {title}\nURL: {record.url}\nTask: {record.query_contexts}\n\n{chunk}"


def chunk_id(history_key: str, url: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{history_key}:{url}:{chunk_index}".encode("utf-8")).hexdigest()
    return f"chunk-{digest}"


def summary_urls(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {normalize_url(clean_text(item.get("url"))) for item in items if isinstance(item, dict) and item.get("url")}


def source_change_status(url: str, added_urls: set[str], changed_urls: set[str]) -> str:
    if url in added_urls:
        return "added"
    if url in changed_urls:
        return "changed"
    return "unchanged"


def delete_source_chunks(collection: Any, history_key: str, url: str) -> None:
    try:
        collection.delete(where={"$and": [{"history_key": history_key}, {"url": url}]})
    except Exception:
        return


def source_content_hash_in_collection(collection: Any, history_key: str, url: str) -> str:
    try:
        result = collection.get(
            where={"$and": [{"history_key": history_key}, {"url": url}]},
            include=["metadatas"],
            limit=1,
        )
    except Exception:
        return ""

    metadatas = result.get("metadatas", []) if isinstance(result, dict) else []
    if not metadatas or not isinstance(metadatas[0], dict):
        return ""
    return clean_text(metadatas[0].get("content_hash"))


def collection_is_empty(collection: Any) -> bool:
    try:
        return collection.count() == 0
    except Exception:
        return False


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in metadata.items():
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int):
            clean[key] = value
        elif isinstance(value, float):
            clean[key] = value
        else:
            clean[key] = clean_text(value)
    return clean


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
