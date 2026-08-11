"""RAG ingestion and indexing helpers backed by LangChain and ChromaDB.

This module is intentionally not an agent. It only converts browser extraction
results into persistent vector-search chunks.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from src.agents.change_detection_agent import hash_text, normalize_url, objective_key
from src.tools.progress import emit_progress
from src.tools.text_utils import clean_text


DEFAULT_COLLECTION_NAME = "research_rag"
DEFAULT_CHROMA_PATH = "data/chroma"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 180
DEFAULT_PARENT_CHUNK_SIZE = 1800
DEFAULT_PARENT_CHUNK_OVERLAP = 240
DEFAULT_PARENT_STORE_NAME = "parent_chunks.sqlite3"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# "auto" prefers CUDA on RunPod, MPS on Apple Silicon, then CPU as fallback.
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_METADATA_SCHEMA_VERSION = 6
TOKEN_PATTERN = re.compile(r"\S+")
SPECIAL_SIGNAL_PATTERN = re.compile(
    r"(?i)("
    r"\\(?:frac|sum|sqrt|top|operatorname)|"
    r"\b(?:equation|formula|softmax|sqrt|formulation"
    r"torch\.|tf\.|keras\.|class\s+\w+|\w+\([^)]*\)|"
    r"benchmark|accuracy|bleu|glue|imagenet|top-1|f1|auc|latency|tokens?/sec)\b|"
    r"[A-Za-z0-9_{}()\\]+\s*=\s*[^=\n]{4,}"
    r")"
)


@dataclass(frozen=True)
class SourceRecord:
    url: str
    title: str
    task_urls: str
    task_ids: str
    query_contexts: str
    source_type: str
    source_quality: str
    content_hash: str
    content: str


class SentenceTransformerEmbeddingFunction:
    """Chroma-compatible embedding function backed by sentence-transformers."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = clean_text(model_name or os.environ.get("RAG_EMBEDDING_MODEL")) or DEFAULT_EMBEDDING_MODEL
        # Set RAG_EMBEDDING_DEVICE=cuda in RunPod to force GPU embeddings.
        self.device = clean_text(device or os.environ.get("RAG_EMBEDDING_DEVICE")) or DEFAULT_EMBEDDING_DEVICE
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

    try:
        import torch
    except ImportError:
        if requested_device and requested_device not in {"auto", "cpu"}:
            raise RuntimeError(f"RAG embedding device '{requested_device}' requires torch, but torch is not installed.")
        return "cpu"

    if requested_device and requested_device != "auto":
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RAG_EMBEDDING_DEVICE=cuda was requested, but CUDA is not available to PyTorch.")
        if requested_device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("RAG_EMBEDDING_DEVICE=mps was requested, but Apple MPS is not available to PyTorch.")
        return requested_device

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

    emit_progress(
        "tool_called",
        "Indexing research content into ChromaDB",
        agent="change_detection",
        tool="chromadb",
        metadata={"chroma_path": str(chroma_path), "collection_name": collection_name},
    )
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
    indexed_parent_chunk_count = 0
    deleted_changed_source_count = 0
    skipped_source_count = 0

    for record in records:
        change_status = source_change_status(record.url, added_urls, changed_urls)
        existing_metadata_schema_version = source_metadata_schema_version_in_collection(collection, history_key, record.url)
        metadata_is_stale = existing_metadata_schema_version < DEFAULT_METADATA_SCHEMA_VERSION
        existing_content_hash = source_content_hash_in_collection(collection, history_key, record.url)
        indexed_content_changed = bool(existing_content_hash and existing_content_hash != record.content_hash)
        if not index_all and change_status == "unchanged" and not metadata_is_stale and not indexed_content_changed:
            skipped_source_count += 1
            continue

        if existing_content_hash == record.content_hash and not metadata_is_stale:
            skipped_source_count += 1
            continue

        source_document = build_langchain_document(record, objective, history_key, change_status if not index_all else "indexed")
        chunk_documents = split_document(source_document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunk_documents:
            skipped_source_count += 1
            continue

        if existing_content_hash and (existing_content_hash != record.content_hash or metadata_is_stale):
            delete_source_chunks(collection, history_key, record.url)
            delete_source_parent_chunks(chroma_path, history_key, record.url)
            if existing_content_hash != record.content_hash:
                deleted_changed_source_count += 1

        indexed_parent_chunk_count += upsert_parent_chunks(chroma_path, parent_rows_from_documents(chunk_documents))
        ids = []
        documents = []
        metadatas = []
        for chunk_index, chunk_document_item in enumerate(chunk_documents):
            ids.append(chunk_id(history_key, record.url, chunk_index))
            documents.append(chunk_document_item.page_content)
            metadata = strip_parent_content_metadata(chunk_document_item.metadata)
            metadatas.append(
                clean_metadata(
                    {
                        **metadata,
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunk_documents),
                    }
                )
            )

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        emit_progress(
            "tool_called",
            "Embedded and upserted source chunks",
            agent="change_detection",
            tool="sentence-transformers/chromadb",
            metadata={"url": record.url, "chunks": len(chunk_documents)},
        )
        indexed_source_count += 1
        indexed_chunk_count += len(chunk_documents)

    return {
        "status": "success",
        "chroma_path": str(chroma_path),
        "collection_name": collection_name,
        "history_key": history_key,
        "indexed_sources": indexed_source_count,
        "indexed_chunks": indexed_chunk_count,
        "indexed_parent_chunks": indexed_parent_chunk_count,
        "parent_store_path": str(parent_store_path(chroma_path)),
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
                "source_url": record.url,
                "task_urls": record.task_urls,
                "title": record.title,
                "task_ids": record.task_ids,
                "query_contexts": record.query_contexts,
                "source_type": record.source_type,
                "source_quality": record.source_quality,
                "content_hash": record.content_hash,
                "metadata_schema_version": DEFAULT_METADATA_SCHEMA_VERSION,
                "change_status": change_status,
            }
        ),
    )


def split_document(document: Any, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Any]:
    Document, _ = langchain_ingestion_classes()
    max_tokens = max(100, chunk_size)
    overlap_tokens = max(0, min(chunk_overlap, max_tokens // 2))
    parent_chunks = parent_context_chunks(document.page_content)
    chunks = token_aware_chunks(document.page_content, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    chunks = add_inter_chunk_overlap(chunks, overlap_tokens=overlap_tokens)
    regular_documents = [
        Document(
            page_content=chunk,
            metadata={
                **document.metadata,
                **parent_child_metadata(chunk, parent_chunks, document.metadata),
                "chunking_strategy": "token_aware_v2",
                "chunk_kind": "regular",
                "token_count": token_count(chunk),
                **chunk_signal_metadata(chunk),
            },
        )
        for chunk in chunks
    ]
    special_documents = [
        Document(
            page_content=chunk,
            metadata={
                **document.metadata,
                **parent_child_metadata(chunk, parent_chunks, document.metadata),
                "chunking_strategy": "token_aware_v2",
                "chunk_kind": "signal",
                "token_count": token_count(chunk),
                **chunk_signal_metadata(chunk),
            },
        )
        for chunk in special_signal_chunks(document.page_content, max_tokens=max(120, max_tokens // 2))
    ]
    return dedupe_documents([*regular_documents, *special_documents])


def parent_context_chunks(text: str) -> list[str]:
    """Build larger parent contexts that child chunks can expand to after retrieval."""

    chunks = token_aware_chunks(
        text,
        max_tokens=DEFAULT_PARENT_CHUNK_SIZE,
        overlap_tokens=DEFAULT_PARENT_CHUNK_OVERLAP,
    )
    return chunks or ([clean_text(text)] if clean_text(text) else [])


def parent_child_metadata(chunk: str, parent_chunks: Sequence[str], metadata: dict[str, Any]) -> dict[str, Any]:
    parent_index, parent_content = best_parent_context(chunk, parent_chunks)
    history_key = clean_text(metadata.get("history_key")) if isinstance(metadata, dict) else ""
    url = clean_text(metadata.get("url")) if isinstance(metadata, dict) else ""
    parent_digest = hashlib.sha256(f"{history_key}:{url}:{parent_index}:{parent_content}".encode("utf-8")).hexdigest()[:16]
    return {
        "parent_id": f"parent-{parent_digest}",
        "parent_index": parent_index,
        "parent_token_count": token_count(parent_content),
        "parent_content": parent_content,
    }


def strip_parent_content_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in (metadata or {}).items() if key != "parent_content"}


def parent_rows_from_documents(documents: Sequence[Any]) -> list[dict[str, Any]]:
    rows_by_id = {}
    for document in documents:
        metadata = getattr(document, "metadata", {}) if document is not None else {}
        if not isinstance(metadata, dict):
            continue
        parent_id = clean_text(metadata.get("parent_id"))
        parent_content = clean_text(metadata.get("parent_content"))
        if not parent_id or not parent_content:
            continue
        rows_by_id[parent_id] = {
            "parent_id": parent_id,
            "history_key": clean_text(metadata.get("history_key")),
            "url": clean_text(metadata.get("url")),
            "parent_index": to_int(metadata.get("parent_index"), 0),
            "token_count": to_int(metadata.get("parent_token_count"), token_count(parent_content)),
            "content_hash": hash_text(parent_content),
            "content": parent_content,
        }
    return list(rows_by_id.values())


def parent_store_path(chroma_path: Union[str, Path]) -> Path:
    path = Path(chroma_path)
    path.mkdir(parents=True, exist_ok=True)
    return path / DEFAULT_PARENT_STORE_NAME


def parent_store_connection(chroma_path: Union[str, Path]) -> sqlite3.Connection:
    connection = sqlite3.connect(parent_store_path(chroma_path))
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_id TEXT PRIMARY KEY,
            history_key TEXT NOT NULL,
            url TEXT NOT NULL,
            parent_index INTEGER NOT NULL,
            token_count INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            content TEXT NOT NULL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute("CREATE INDEX IF NOT EXISTS idx_parent_chunks_source ON parent_chunks(history_key, url)")
    return connection


def upsert_parent_chunks(chroma_path: Union[str, Path], rows: Sequence[dict[str, Any]]) -> int:
    clean_rows = [row for row in rows if clean_text(row.get("parent_id")) and clean_text(row.get("content"))]
    if not clean_rows:
        return 0
    with parent_store_connection(chroma_path) as connection:
        connection.executemany(
            """
            INSERT INTO parent_chunks (
                parent_id, history_key, url, parent_index, token_count, content_hash, content, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(parent_id) DO UPDATE SET
                history_key=excluded.history_key,
                url=excluded.url,
                parent_index=excluded.parent_index,
                token_count=excluded.token_count,
                content_hash=excluded.content_hash,
                content=excluded.content,
                updated_at=CURRENT_TIMESTAMP
            """,
            [
                (
                    clean_text(row.get("parent_id")),
                    clean_text(row.get("history_key")),
                    clean_text(row.get("url")),
                    to_int(row.get("parent_index"), 0),
                    to_int(row.get("token_count"), token_count(row.get("content"))),
                    clean_text(row.get("content_hash")) or hash_text(clean_text(row.get("content"))),
                    clean_text(row.get("content")),
                )
                for row in clean_rows
            ],
        )
    return len(clean_rows)


def parent_content_for_id(chroma_path: Union[str, Path], parent_id: str) -> str:
    parent_id = clean_text(parent_id)
    if not parent_id:
        return ""
    try:
        with parent_store_connection(chroma_path) as connection:
            row = connection.execute(
                "SELECT content FROM parent_chunks WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
    except sqlite3.Error:
        return ""
    return clean_text(row[0]) if row else ""


def best_parent_context(chunk: str, parent_chunks: Sequence[str]) -> tuple[int, str]:
    if not parent_chunks:
        return 0, clean_text(chunk)
    chunk_terms = set(TOKEN_PATTERN.findall(clean_text(chunk).lower()))
    if not chunk_terms:
        return 0, clean_text(parent_chunks[0])
    scored = [
        (len(chunk_terms & set(TOKEN_PATTERN.findall(clean_text(parent).lower()))), index, parent)
        for index, parent in enumerate(parent_chunks)
    ]
    _, index, parent = max(scored, key=lambda item: (item[0], -item[1]))
    return index, clean_text(parent)


def token_aware_chunks(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Create ordered token-bounded chunks while preserving small text blocks."""
    blocks = text_blocks(text)
    chunks = []
    current_blocks = []
    current_tokens = 0

    for block in blocks:
        block_tokens = token_count(block)
        if block_tokens <= 0:
            continue

        if block_tokens > max_tokens:
            if current_blocks:
                chunks.append(join_blocks(current_blocks))
                current_blocks = []
                current_tokens = 0
            chunks.extend(split_large_block(block, max_tokens=max_tokens, overlap_tokens=overlap_tokens))
            continue

        if current_blocks and current_tokens + block_tokens > max_tokens:
            chunks.append(join_blocks(current_blocks))
            current_blocks = [block]
            current_tokens = block_tokens
            continue

        current_blocks.append(block)
        current_tokens += block_tokens

    if current_blocks:
        chunks.append(join_blocks(current_blocks))

    return [chunk for chunk in chunks if clean_text(chunk)]


def add_inter_chunk_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = " ".join(TOKEN_PATTERN.findall(previous)[-overlap_tokens:])
        overlapped.append(join_blocks([prefix, current]) if prefix else current)
    return overlapped


def special_signal_chunks(text: str, max_tokens: int, context_lines: int = 3) -> list[str]:
    lines = [line for line in str(text or "").splitlines() if clean_text(line)]
    chunks = []
    for index, line in enumerate(lines):
        if not SPECIAL_SIGNAL_PATTERN.search(line):
            continue
        start = max(0, index - context_lines)
        end = min(len(lines), index + context_lines + 1)
        chunk = clean_text("\n".join(lines[start:end]))
        if token_count(chunk) > max_tokens:
            chunk = " ".join(TOKEN_PATTERN.findall(chunk)[:max_tokens])
        if chunk:
            chunks.append(chunk)
    return dedupe_preserve_order(chunks)


def chunk_signal_metadata(chunk: str) -> dict[str, Any]:
    value = str(chunk or "")
    lowered = value.lower()
    return {
        "has_formula_signal": bool(
            re.search(r"\\(?:frac|sum|sqrt|top|operatorname)|\b(?:softmax|sqrt)\s*\(", value, flags=re.I)
            or re.search(r"[A-Za-z0-9_{}()\\]+\s*=\s*[^=\n]{4,}", value)
        ),
        "has_api_signal": bool(re.search(r"\b(?:torch\.|tf\.|keras\.)[A-Za-z0-9_.]+", value)),
        "has_benchmark_signal": any(
            term in lowered
            for term in ("benchmark", "accuracy", "bleu", "glue", "imagenet", "top-1", "f1", "auc")
        ),
    }


def dedupe_documents(documents: list[Any]) -> list[Any]:
    deduped = []
    seen = set()
    for document in documents:
        metadata = document.metadata if isinstance(document.metadata, dict) else {}
        key = f"{clean_text(metadata.get('chunk_kind'))}:{clean_text(document.page_content)}"
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    deduped = []
    seen = set()
    for item in items:
        value = clean_text(item)
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def text_blocks(text: str) -> list[str]:
    """Split text into coarse blocks without requiring section detection."""
    raw_blocks = re.split(r"\n\s*\n+", str(text or ""))
    blocks = [clean_text(block) for block in raw_blocks if clean_text(block)]
    return blocks or ([clean_text(text)] if clean_text(text) else [])


def split_large_block(block: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    tokens = TOKEN_PATTERN.findall(clean_text(block))
    if not tokens:
        return []

    chunks = []
    start = 0
    step = max(1, max_tokens - overlap_tokens)
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        if window:
            chunks.append(" ".join(window))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return chunks


def join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(clean_text(block) for block in blocks if clean_text(block))


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(clean_text(text)))


def langchain_ingestion_classes() -> tuple[Any, Any]:
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter
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
        task_url = normalize_http_url(result.get("task_url"))
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
                    "task_urls": [],
                    "task_ids": [],
                    "query_contexts": [],
                    "source_type": clean_text(source.get("source_type")),
                    "source_quality": clean_text(source.get("source_quality")),
                    "content": content,
                },
            )
            if task_id and task_id not in record["task_ids"]:
                record["task_ids"].append(task_id)
            if task_url and task_url not in record["task_urls"]:
                record["task_urls"].append(task_url)
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
            task_urls=" | ".join(record["task_urls"]),
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


def delete_source_parent_chunks(chroma_path: Union[str, Path], history_key: str, url: str) -> None:
    try:
        with parent_store_connection(chroma_path) as connection:
            connection.execute(
                "DELETE FROM parent_chunks WHERE history_key = ? AND url = ?",
                (clean_text(history_key), clean_text(url)),
            )
    except sqlite3.Error:
        return


def source_content_hash_in_collection(collection: Any, history_key: str, url: str) -> str:
    metadata = source_metadata_in_collection(collection, history_key, url)
    return clean_text(metadata.get("content_hash")) if metadata else ""


def source_metadata_schema_version_in_collection(collection: Any, history_key: str, url: str) -> int:
    metadata = source_metadata_in_collection(collection, history_key, url)
    return to_int(metadata.get("metadata_schema_version") if metadata else None, 0)


def source_metadata_in_collection(collection: Any, history_key: str, url: str) -> dict[str, Any]:
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
        return {}
    return metadatas[0]


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


def normalize_http_url(value: Any) -> str:
    url = normalize_url(clean_text(value))
    return url if url.startswith(("http://", "https://")) else ""
