"""RAG ingestion and indexing helpers backed by LangChain and ChromaDB.

This module is intentionally not an agent. It only converts browser extraction
results into persistent vector-search chunks.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sqlite3
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from src.agents.change_detection_agent import hash_text, normalize_url, objective_key
from src.tools.progress import emit_progress
from src.tools.text_utils import clean_text


DEFAULT_COLLECTION_NAME = "research_rag_bge_large"
DEFAULT_CHROMA_PATH = "data/chroma"
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_PARENT_CHUNK_SIZE = 4000
DEFAULT_PARENT_CHUNK_OVERLAP = 400
DEFAULT_PARENT_STORE_NAME = "parent_chunks.sqlite3"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# "auto" prefers CUDA on RunPod, MPS on Apple Silicon, then CPU as fallback.
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_CUDA_EMBEDDING_BATCH_SIZE = 4
DEFAULT_CHROMA_UPSERT_BATCH_SIZE = 8
DEFAULT_METADATA_SCHEMA_VERSION = 9
HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
TOKEN_WRAPPER_QUOTES = "\"'\u201c\u201d\u2018\u2019"
TOKEN_PATTERN = re.compile(r"\S+")
SECTION_HEADING_PATTERN = re.compile(
    r"(?i)^\s*(?:#{1,6}\s+|\d+(?:\.\d+)*[.)]\s+)?"
    r"(abstract|introduction|background|related work|method|methods|methodology|"
    r"model|architecture|approach|algorithm|implementation|experiments?|evaluation|"
    r"results?|discussion|analysis|limitations?|conclusion|references?|appendix)\b"
)
NOISE_LINE_PATTERN = re.compile(
    r"(?i)\b("
    r"accept all|cookie|privacy policy|terms of use|all rights reserved|"
    r"subscribe|newsletter|sign in|sign up|log in|skip to|share this|"
    r"advertisement|sponsored|enable javascript|table of contents|"
    r"download pdf|view pdf|back to top"
    r")\b"
)
SPECIAL_SIGNAL_PATTERN = re.compile(
    r"(?i)("
    r"\\(?:frac|sum|sqrt|top|operatorname)|"
    r"\b(?:equation|formula|softmax|sqrt|formulation"
    r"torch\.|tf\.|keras\.|class\s+\w+|\w+\([^)]*\)|"
    r"benchmark|accuracy|bleu|glue|imagenet|top-1|f1|auc|latency|tokens?/sec)\b|"
    r"[A-Za-z0-9_{}()\\]+\s*=\s*[^=\n]{4,}"
    r")"
)
STORAGE_TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2211": "sum",
        "\u221a": "sqrt",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u2248": "~=",
        "\u00d7": "x",
        "\u00b2": "^2",
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03bb": "lambda",
        "\u03c3": "sigma",
    }
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
    source_authority: str
    content_noise_score: float
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
        batch_size = embedding_batch_size(self.device)
        while True:
            try:
                embeddings = model.encode(
                    input,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return embeddings.tolist()
            except RuntimeError as error:
                if not is_cuda_oom_error(error) or batch_size <= 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"[rag_index] CUDA OOM while embedding; retrying with batch_size={batch_size}")
                clear_embedding_model_memory()
            finally:
                clear_embedding_model_memory()

    def name(self) -> str:
        """Stable Chroma embedding function name used when reopening collections."""
        safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
        return f"sentence-transformers_{safe_model_name}"

    def model(self) -> Any:
        if self._model is not None:
            return self._model
        normalize_huggingface_token_env()
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

    def close(self) -> None:
        """Release the loaded embedding model between full graph runs."""
        if self._model is not None:
            try:
                if hasattr(self._model, "to"):
                    self._model.to("cpu")
            except Exception:
                pass
        self._model = None
        clear_embedding_model_memory()


def normalize_huggingface_token_env() -> None:
    """Normalize copied Hugging Face tokens before HTTP requests use them."""

    for name in HF_TOKEN_ENV_VARS:
        value = os.environ.get(name)
        if value is None:
            continue
        cleaned = str(value).strip().strip(TOKEN_WRAPPER_QUOTES)
        try:
            cleaned.encode("latin-1")
        except UnicodeEncodeError as error:
            unsafe = ", ".join(f"U+{ord(char):04X} {repr(char)}" for char in latin1_unsafe_characters(cleaned)[:8])
            raise RuntimeError(
                f"{name} contains characters that cannot be sent in HTTP headers: {unsafe}. "
                f"Re-export it with plain ASCII characters, for example: export {name}=hf_..."
            ) from error
        if cleaned != value:
            os.environ[name] = cleaned
            print(f"[rag_index] normalized {name}; removed surrounding quotes/whitespace")


def clear_embedding_model_memory() -> None:
    """Best-effort cleanup for torch-backed embedding models."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_available and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


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


def embedding_batch_size(device: str) -> int:
    """Return a conservative encode batch size for the selected device."""

    if os.environ.get("RAG_EMBEDDING_BATCH_SIZE"):
        return max(1, to_int(os.environ.get("RAG_EMBEDDING_BATCH_SIZE"), DEFAULT_EMBEDDING_BATCH_SIZE))
    if clean_text(device).lower().startswith("cuda"):
        return DEFAULT_CUDA_EMBEDDING_BATCH_SIZE
    return DEFAULT_EMBEDDING_BATCH_SIZE


def upsert_batch_size() -> int:
    return max(1, to_int(os.environ.get("RAG_CHROMA_UPSERT_BATCH_SIZE"), DEFAULT_CHROMA_UPSERT_BATCH_SIZE))


def upsert_collection_batches(
    collection: Any,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
) -> int:
    """Upsert Chroma chunks in small batches to avoid embedding-time GPU spikes."""

    total = 0
    batch_size = upsert_batch_size()
    index = 0
    while index < len(ids):
        end = min(len(ids), index + batch_size)
        batch_documents = []
        batch_metadatas = []
        try:
            batch_documents = [storage_safe_text(document) for document in documents[index:end]]
            batch_metadatas = [clean_metadata(metadata) for metadata in metadatas[index:end]]
            collection.upsert(
                ids=list(ids[index:end]),
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            total += end - index
            index = end
            clear_embedding_model_memory()
        except RuntimeError as error:
            if not is_cuda_oom_error(error) or batch_size <= 1:
                raise
            failed_batch_size = end - index
            batch_size = max(1, min(batch_size // 2, failed_batch_size // 2))
            print(f"[rag_index] CUDA OOM during Chroma upsert; retrying with batch_size={batch_size}")
            clear_embedding_model_memory()
        except Exception as error:
            print_indexing_error(
                error,
                stage="chroma_upsert",
                ids=list(ids[index:end]),
                documents=batch_documents or documents[index:end],
                metadatas=batch_metadatas or metadatas[index:end],
            )
            raise
    return total


def is_cuda_oom_error(error: BaseException) -> bool:
    message = clean_text(error).lower()
    return "cuda out of memory" in message or ("out of memory" in message and "cuda" in message)


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
    embedding_function = SentenceTransformerEmbeddingFunction()
    collection = get_collection(chroma_path, collection_name, embedding_function=embedding_function)
    change_detection = change_detection if isinstance(change_detection, dict) else {}
    research_plan = research_plan if isinstance(research_plan, dict) else {}
    objective = clean_text(change_detection.get("objective") or research_plan.get("objective"))
    history_key = clean_text(change_detection.get("history_key")) or objective_key(objective, research_plan)

    records: list[SourceRecord] = []
    added_urls = set()
    changed_urls = set()
    removed_urls = set()
    index_all = False
    indexed_source_count = 0
    indexed_chunk_count = 0
    indexed_parent_chunk_count = 0
    deleted_changed_source_count = 0
    skipped_source_count = 0
    current_stage = "source_records"
    current_record: SourceRecord | None = None

    try:
        current_stage = "source_records"
        records = list(source_records(browser_results))
        added_urls = summary_urls(change_detection.get("added_sources"))
        changed_urls = summary_urls(change_detection.get("changed_sources"))
        removed_urls = summary_urls(change_detection.get("removed_sources"))
        index_all = bool(change_detection.get("first_run")) or collection_is_empty(collection)

        for record in records:
            current_record = record
            change_status = source_change_status(record.url, added_urls, changed_urls)
            current_stage = "metadata_lookup"
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

            current_stage = "build_document"
            source_document = build_langchain_document(record, objective, history_key, change_status if not index_all else "indexed")
            current_stage = "split_document"
            chunk_documents = split_document(source_document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunk_documents:
                skipped_source_count += 1
                continue

            if existing_content_hash and (existing_content_hash != record.content_hash or metadata_is_stale):
                current_stage = "delete_stale_chunks"
                delete_source_chunks(collection, history_key, record.url)
                delete_source_parent_chunks(chroma_path, history_key, record.url)
                if existing_content_hash != record.content_hash:
                    deleted_changed_source_count += 1

            current_stage = "parent_upsert"
            indexed_parent_chunk_count += upsert_parent_chunks(chroma_path, parent_rows_from_documents(chunk_documents))
            current_stage = "prepare_child_chunks"
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

            current_stage = "child_upsert"
            upserted_chunks = upsert_collection_batches(collection, ids, documents, metadatas)
            emit_progress(
                "tool_called",
                "Embedded and upserted source chunks",
                agent="change_detection",
                tool="sentence-transformers/chromadb",
                metadata={"url": record.url, "chunks": upserted_chunks},
            )
            indexed_source_count += 1
            indexed_chunk_count += upserted_chunks
    except Exception as error:
        print_indexing_error(
            error,
            stage=current_stage,
            record=current_record,
        )
        raise
    finally:
        embedding_function.close()

    stored_chunk_count = collection_count(collection)
    return {
        "status": "success",
        "chroma_path": str(chroma_path),
        "collection_name": collection_name,
        "history_key": history_key,
        "indexed_sources": indexed_source_count,
        "indexed_chunks": indexed_chunk_count,
        "stored_chunks": stored_chunk_count,
        "indexed_parent_chunks": indexed_parent_chunk_count,
        "parent_store_path": str(parent_store_path(chroma_path)),
        "deleted_changed_sources": deleted_changed_source_count,
        "preserved_removed_sources": len(removed_urls),
        "skipped_sources": skipped_source_count,
        "total_sources_seen": len(records),
    }


def get_collection(
    chroma_path: Union[str, Path],
    collection_name: str,
    embedding_function: Optional[SentenceTransformerEmbeddingFunction] = None,
):
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
        embedding_function=embedding_function or SentenceTransformerEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def build_langchain_document(record: SourceRecord, objective: str, history_key: str, change_status: str) -> Any:
    Document, _ = langchain_ingestion_classes()
    return Document(
        page_content=chunk_document(record, clean_document_text(record.content)),
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
                "source_authority": record.source_authority,
                "content_noise_score": record.content_noise_score,
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
    content = clean_document_text(document.page_content)
    parent_chunks = parent_context_chunks(content)
    chunks = token_aware_chunks(content, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
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
        for chunk in special_signal_chunks(content, max_tokens=max(120, max_tokens // 2))
    ]
    table_documents = [
        Document(
            page_content=chunk,
            metadata={
                **document.metadata,
                **parent_child_metadata(chunk, parent_chunks, document.metadata),
                "chunking_strategy": "table_json_v1",
                "chunk_kind": "table",
                "token_count": token_count(chunk),
                **metadata,
                **chunk_signal_metadata(chunk),
            },
        )
        for chunk, metadata in table_json_chunks(content)
    ]
    return dedupe_documents([*regular_documents, *special_documents, *table_documents])


def parent_context_chunks(text: str) -> list[str]:
    """Build larger parent contexts that child chunks can expand to after retrieval."""

    chunks = structure_aware_chunks(
        clean_document_text(text),
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
        parent_content = storage_safe_text(metadata.get("parent_content"))
        if not parent_id or not parent_content:
            continue
        rows_by_id[parent_id] = {
            "parent_id": parent_id,
            "history_key": storage_safe_text(metadata.get("history_key")),
            "url": storage_safe_text(metadata.get("url")),
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
                    storage_safe_text(row.get("history_key")),
                    storage_safe_text(row.get("url")),
                    to_int(row.get("parent_index"), 0),
                    to_int(row.get("token_count"), token_count(row.get("content"))),
                    clean_text(row.get("content_hash")) or hash_text(storage_safe_text(row.get("content"))),
                    storage_safe_text(row.get("content")),
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
    return storage_safe_text(row[0]) if row else ""


def best_parent_context(chunk: str, parent_chunks: Sequence[str]) -> tuple[int, str]:
    if not parent_chunks:
        return 0, storage_safe_text(chunk)
    chunk_terms = set(TOKEN_PATTERN.findall(clean_text(chunk).lower()))
    if not chunk_terms:
        return 0, storage_safe_text(parent_chunks[0])
    scored = [
        (len(chunk_terms & set(TOKEN_PATTERN.findall(clean_text(parent).lower()))), index, parent)
        for index, parent in enumerate(parent_chunks)
    ]
    _, index, parent = max(scored, key=lambda item: (item[0], -item[1]))
    return index, storage_safe_text(parent)


def clean_document_text(text: Any) -> str:
    """Remove common web/PDF extraction noise before chunking."""

    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"https?://\S+\.(?:png|jpg|jpeg|gif|svg|webp)\S*", "", value, flags=re.I)
    value = re.sub(r"[ \t]+", " ", value)
    lines = []
    seen_counts: dict[str, int] = {}
    for raw_line in value.splitlines():
        line = clean_text(raw_line)
        if not line:
            lines.append("")
            continue
        normalized = re.sub(r"\W+", " ", line.lower()).strip()
        if should_drop_noise_line(line, normalized, seen_counts):
            continue
        seen_counts[normalized] = seen_counts.get(normalized, 0) + 1
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return storage_safe_text(cleaned.strip())


def should_drop_noise_line(line: str, normalized: str, seen_counts: dict[str, int]) -> bool:
    if not normalized:
        return False
    if NOISE_LINE_PATTERN.search(line) and token_count(line) <= 18:
        return True
    if normalized.startswith(("home ", "menu ", "navigation ", "search ")):
        return True
    if len(normalized) <= 4 and seen_counts.get(normalized, 0) >= 1:
        return True
    if seen_counts.get(normalized, 0) >= 2 and token_count(line) <= 24:
        return True
    return False


def structure_aware_chunks(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack section-like blocks into parent chunks while preserving boundaries."""

    blocks = structured_text_blocks(text)
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
    return add_inter_chunk_overlap([chunk for chunk in chunks if clean_text(chunk)], overlap_tokens=overlap_tokens)


def structured_text_blocks(text: str) -> list[str]:
    """Split text at headings while keeping tables, equations, and code fences intact."""

    sections = []
    current = []
    in_fence = False
    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        starts_new_section = bool(current and not in_fence and is_structure_heading(stripped))
        if starts_new_section:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    blocks = []
    for section in sections:
        section_blocks = text_blocks(section)
        if token_count(section) <= DEFAULT_PARENT_CHUNK_SIZE:
            blocks.append(section)
        else:
            blocks.extend(section_blocks)
    return [block for block in blocks if clean_text(block)]


def is_structure_heading(line: str) -> bool:
    if not line or len(line) > 140:
        return False
    if line.startswith(("#", "##")):
        return True
    if SECTION_HEADING_PATTERN.search(line):
        return True
    if re.match(r"^\d+(?:\.\d+)*[.)]\s+[A-Z][A-Za-z0-9 ,:;()/_-]{3,}$", line):
        return True
    return False


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


def table_json_chunks(text: str) -> list[tuple[str, dict[str, Any]]]:
    chunks = []
    for table_index, rows in enumerate(extract_table_rows(text), start=1):
        headers = unique_table_headers(rows[0])
        data_rows = rows[1:] if len(rows) > 1 else []
        payload = {
            "table_index": table_index,
            "headers": headers,
            "rows": data_rows,
            "records": [table_record(headers, row) for row in data_rows],
        }
        chunk = storage_safe_text("Table data JSON:\n" + json.dumps(payload, ensure_ascii=True, indent=2))
        chunks.append(
            (
                chunk,
                {
                    "has_table_signal": True,
                    "table_index": table_index,
                    "table_row_count": len(data_rows),
                    "table_column_count": max(len(row) for row in rows),
                },
            )
        )
    return chunks


def extract_table_rows(text: str) -> list[list[list[str]]]:
    tables = []
    current = []
    for raw_line in str(text or "").splitlines():
        cells = table_row_cells(raw_line)
        if cells:
            if not table_separator_row(cells):
                current.append(cells)
            continue
        if len(current) >= 2:
            tables.append(current)
        current = []
    if len(current) >= 2:
        tables.append(current)
    return tables


def table_row_cells(line: str) -> list[str]:
    if "|" not in str(line):
        return []
    cells = [storage_safe_text(cell) for cell in str(line).strip().strip("|").split("|")]
    cells = [cell for cell in cells if cell]
    return cells if len(cells) >= 2 else []


def table_separator_row(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells)


def unique_table_headers(cells: Sequence[str]) -> list[str]:
    headers = []
    seen = {}
    for index, cell in enumerate(cells, start=1):
        header = storage_safe_text(cell) or f"column_{index}"
        key = header.lower()
        seen[key] = seen.get(key, 0) + 1
        headers.append(header if seen[key] == 1 else f"{header}_{seen[key]}")
    return headers


def table_record(headers: Sequence[str], row: Sequence[str]) -> dict[str, str]:
    return {header: storage_safe_text(row[index]) if index < len(row) else "" for index, header in enumerate(headers)}


def chunk_signal_metadata(chunk: str) -> dict[str, Any]:
    value = str(chunk or "")
    lowered = value.lower()
    return {
        "has_formula_signal": bool(
            re.search(r"\\(?:frac|sum|sqrt|top|operatorname)|\b(?:softmax|sqrt)\s*\(", value, flags=re.I)
            or re.search(r"[A-Za-z0-9_{}()\\]+\s*=\s*[^=\n]{4,}", value)
        ),
        "has_table_signal": bool("Table data JSON:" in value or re.search(r"\|[^|\n]+\|", value)),
        "has_api_signal": bool(re.search(r"\b(?:torch\.|tf\.|keras\.)[A-Za-z0-9_.]+", value)),
        "has_benchmark_signal": any(
            term in lowered
            for term in ("benchmark", "accuracy", "bleu", "glue", "imagenet", "top-1", "f1", "auc", "score")
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
        task_id = storage_safe_text(result.get("task_id"))
        task_url = normalize_http_url(result.get("task_url"))
        query_context = storage_safe_text(result.get("query_context"))
        for source in result.get("sources", []):
            if not isinstance(source, dict):
                continue
            url = normalize_url(clean_text(source.get("url")))
            content = clean_document_text(source.get("full_content") or source.get("content_preview"))
            if not url or not content:
                continue

            record = grouped.setdefault(
                url,
                {
                    "url": url,
                    "title": storage_safe_text(source.get("title")),
                    "task_urls": [],
                    "task_ids": [],
                    "query_contexts": [],
                    "source_type": storage_safe_text(source.get("source_type")),
                    "source_quality": storage_safe_text(source.get("source_quality")),
                    "source_authority": storage_safe_text(source.get("source_authority")),
                    "content_noise_score": to_float(source.get("content_noise_score"), 0.0),
                    "content": content,
                },
            )
            if task_id and task_id not in record["task_ids"]:
                record["task_ids"].append(task_id)
            if task_url and task_url not in record["task_urls"]:
                record["task_urls"].append(task_url)
            if query_context and query_context not in record["query_contexts"]:
                record["query_contexts"].append(query_context)

            if source_quality_rank(source.get("source_quality")) > source_quality_rank(record["source_quality"]):
                record["source_quality"] = storage_safe_text(source.get("source_quality"))
            if source_authority_rank(source.get("source_authority")) > source_authority_rank(record["source_authority"]):
                record["source_authority"] = storage_safe_text(source.get("source_authority"))
            if len(content) > len(record["content"]):
                record["content"] = content
                record["title"] = storage_safe_text(source.get("title")) or record["title"]
                record["source_type"] = storage_safe_text(source.get("source_type")) or record["source_type"]
                record["content_noise_score"] = to_float(source.get("content_noise_score"), record["content_noise_score"])

    for record in grouped.values():
        content = storage_safe_text(record["content"])
        yield SourceRecord(
            url=record["url"],
            title=storage_safe_text(record["title"]),
            task_urls=storage_safe_text(" | ".join(record["task_urls"])),
            task_ids=storage_safe_text(", ".join(record["task_ids"])),
            query_contexts=storage_safe_text(" | ".join(record["query_contexts"])),
            source_type=storage_safe_text(record["source_type"]),
            source_quality=storage_safe_text(record["source_quality"]),
            source_authority=storage_safe_text(record["source_authority"]),
            content_noise_score=record["content_noise_score"],
            content_hash=hash_text(content),
            content=content,
        )


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    Document, _ = langchain_ingestion_classes()
    document = Document(page_content=clean_text(text), metadata={})
    return [chunk.page_content for chunk in split_document(document, chunk_size=chunk_size, chunk_overlap=overlap)]


def chunk_document(record: SourceRecord, chunk: str) -> str:
    title = storage_safe_text(record.title or record.url)
    return storage_safe_text(f"Source: {title}\nURL: {record.url}\nTask: {record.query_contexts}\n\n{chunk}")


def chunk_id(history_key: str, url: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{history_key}:{url}:{chunk_index}".encode("utf-8")).hexdigest()
    return f"chunk-{digest}"


def source_quality_rank(value: Any) -> int:
    text = clean_text(value).lower()
    if "primary" in text:
        return 5
    if "official" in text:
        return 4
    if "authoritative" in text:
        return 3
    if "useful" in text:
        return 2
    if "weak" in text:
        return 1
    return 0


def source_authority_rank(value: Any) -> int:
    text = clean_text(value).lower()
    ranks = {"primary": 5, "official": 4, "authoritative": 3, "topic_match": 2, "secondary": 1}
    return ranks.get(text, 0)


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


def collection_count(collection: Any) -> int:
    try:
        return int(collection.count())
    except Exception:
        return 0


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
            clean[key] = storage_safe_text(value)
    return clean


def storage_safe_text(value: Any) -> str:
    """Normalize extracted text before handing it to storage/embedding clients."""

    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = raw.translate(STORAGE_TEXT_REPLACEMENTS).encode("latin-1", errors="replace").decode("latin-1")
    if "\n" not in text:
        return clean_text(text)
    lines = [clean_text(line) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def print_indexing_error(
    error: BaseException,
    stage: str,
    record: SourceRecord | None = None,
    ids: Sequence[str] | None = None,
    documents: Sequence[Any] | None = None,
    metadatas: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Print compact diagnostics for indexing failures that are otherwise opaque."""

    print(f"[rag_index] failed during {clean_text(stage) or 'unknown_stage'}: {type(error).__name__}: {error}")
    if record is not None:
        print(f"[rag_index] source url={record.url or '<missing>'} title={record.title[:160] or '<missing>'}")
        print_latin1_diagnostics("record.title", record.title)
        print_latin1_diagnostics("record.query_contexts", record.query_contexts)
        print_latin1_diagnostics("record.content", record.content)
    if ids:
        print(f"[rag_index] batch ids={list(ids)[:5]} count={len(ids)}")
    for index, document in enumerate(list(documents or [])[:3], start=1):
        print_latin1_diagnostics(f"batch.document.{index}", document)
    for index, metadata in enumerate(list(metadatas or [])[:3], start=1):
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            print_latin1_diagnostics(f"batch.metadata.{index}.{key}", value)
    details = "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    if details:
        print(f"[rag_index] traceback:\n{details}")


def print_latin1_diagnostics(label: str, value: Any) -> None:
    unsafe = latin1_unsafe_characters(value)
    if not unsafe:
        return
    preview = clean_text(value)[:220]
    chars = ", ".join(f"U+{ord(char):04X} {repr(char)}" for char in unsafe[:8])
    print(f"[rag_index] latin-1 unsafe in {label}: {chars}; preview={preview}")


def latin1_unsafe_characters(value: Any) -> list[str]:
    seen = set()
    unsafe = []
    for char in str(value or ""):
        try:
            char.encode("latin-1")
        except UnicodeEncodeError:
            if char not in seen:
                seen.add(char)
                unsafe.append(char)
    return unsafe


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_http_url(value: Any) -> str:
    url = normalize_url(clean_text(value))
    return url if url.startswith(("http://", "https://")) else ""
