import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from src.rag.indexing import (
    clean_document_text,
    langchain_ingestion_classes,
    normalize_huggingface_token_env,
    parent_content_for_id,
    parent_context_chunks,
    parent_rows_from_documents,
    source_records,
    split_document,
    storage_safe_text,
    structure_aware_chunks,
    strip_parent_content_metadata,
    upsert_collection_batches,
    upsert_parent_chunks,
)
from src.rag.retrieval import RetrievalResult, expand_parent_context_results, metadata_signal_score


class IndexingChunkingTests(unittest.TestCase):
    def test_clean_document_text_removes_common_web_noise(self):
        text = """Skip to content
Accept all cookies
Useful technical paragraph with an equation y = softmax(x).
Subscribe to our newsletter
Useful technical paragraph with an equation y = softmax(x).
Useful technical paragraph with an equation y = softmax(x).
"""

        cleaned = clean_document_text(text)

        self.assertNotIn("Accept all cookies", cleaned)
        self.assertNotIn("Subscribe to our newsletter", cleaned)
        self.assertEqual(cleaned.count("Useful technical paragraph"), 2)

    def test_structure_aware_chunks_preserve_sections_for_parent_context(self):
        text = """# Abstract
This section summarizes the document.

## Method
The method keeps equations near their explanation.
y = softmax(x)

## Results
The results section keeps benchmark text together.
Accuracy is 91.0%.
"""

        chunks = structure_aware_chunks(text, max_tokens=18, overlap_tokens=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("## Method" in chunk and "y = softmax(x)" in chunk for chunk in chunks))
        self.assertTrue(any("## Results" in chunk and "Accuracy is 91.0%" in chunk for chunk in chunks))

    def test_parent_context_chunks_use_structure_boundaries(self):
        text = "\n\n".join(
            [
                "## Setup\n" + "Setup details. " * 80,
                "## API\n" + "torch.example_call(arg=True). " * 80,
                "## Metrics\n" + "Accuracy is 91.0%. " * 80,
            ]
        )

        parents = parent_context_chunks(text)

        self.assertTrue(parents)
        self.assertTrue(any("## API" in parent for parent in parents))

    def test_split_document_adds_signal_chunks_for_precise_evidence(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content="""Technical paper notes.

The formula is score=softmax(logits/sqrt(width)).

PyTorch exposes torch.nn.MultiheadAttention(embed_dim, num_heads).

ImageNet top-1 accuracy improves by 1.00%.
""",
            metadata={"url": "https://example.com", "history_key": "test"},
        )

        chunks = split_document(document)
        signal_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") == "signal"]

        self.assertTrue(signal_chunks)
        self.assertTrue(any(chunk.metadata.get("has_formula_signal") for chunk in signal_chunks))
        self.assertTrue(any(chunk.metadata.get("has_api_signal") for chunk in signal_chunks))
        self.assertTrue(any(chunk.metadata.get("has_benchmark_signal") for chunk in signal_chunks))

    def test_split_document_adds_table_json_chunk(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content="""Benchmark section.

Model | BLEU | Params
Transformer | 28.4 | 65M
RNN baseline | 24.6 | 80M

The table above should stay intact.
""",
            metadata={"url": "https://example.com/results", "history_key": "test"},
        )

        chunks = split_document(document)
        table_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") == "table"]

        self.assertEqual(len(table_chunks), 1)
        self.assertTrue(table_chunks[0].metadata.get("has_table_signal"))
        self.assertTrue(table_chunks[0].metadata.get("has_benchmark_signal"))
        self.assertEqual(table_chunks[0].metadata.get("table_row_count"), 2)

        payload = json.loads(table_chunks[0].page_content.removeprefix("Table data JSON:\n"))
        self.assertEqual(payload["headers"], ["Model", "BLEU", "Params"])
        self.assertEqual(payload["records"][0]["Model"], "Transformer")

    def test_split_document_sanitizes_table_json_chunk_text(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content='Model | Note\nAlpha | Uses \u201csmart quotes\u201d and O(n\u00b2)',
            metadata={"url": "https://example.com/results", "history_key": "test"},
        )

        chunks = split_document(document)
        table_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") == "table"]

        self.assertEqual(len(table_chunks), 1)
        table_chunks[0].page_content.encode("latin-1")
        payload = json.loads(table_chunks[0].page_content.removeprefix("Table data JSON:\n"))
        self.assertEqual(payload["records"][0]["Note"], 'Uses "smart quotes" and O(n^2)')

    def test_split_document_adds_parent_context_metadata(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content="\n\n".join(f"Section {index} has detailed context about attention formulas." for index in range(80)),
            metadata={"url": "https://example.com", "history_key": "test"},
        )

        chunks = split_document(document, chunk_size=120, chunk_overlap=20)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.metadata.get("parent_id") for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("parent_content") for chunk in chunks))
        self.assertTrue(any(chunk.metadata.get("parent_token_count", 0) > chunk.metadata.get("token_count", 0) for chunk in chunks))

    def test_parent_chunks_are_stored_in_sqlite_and_stripped_from_child_metadata(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content="\n\n".join(f"Section {index} has detailed context about attention formulas." for index in range(80)),
            metadata={"url": "https://example.com", "history_key": "test"},
        )
        chunks = split_document(document, chunk_size=120, chunk_overlap=20)
        parent_id = chunks[0].metadata["parent_id"]

        with tempfile.TemporaryDirectory() as tmpdir:
            stored = upsert_parent_chunks(tmpdir, parent_rows_from_documents(chunks))

            self.assertGreater(stored, 0)
            self.assertIn("attention formulas", parent_content_for_id(tmpdir, parent_id))
            self.assertNotIn("parent_content", strip_parent_content_metadata(chunks[0].metadata))

    def test_parent_rows_sanitize_unicode_content_before_storage(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content='\u201cAttention\u201d with O(n\u00b2) context. ' * 80,
            metadata={"url": "https://example.com", "history_key": "test"},
        )
        chunks = split_document(document, chunk_size=120, chunk_overlap=20)

        rows = parent_rows_from_documents(chunks)

        self.assertTrue(rows)
        rows[0]["content"].encode("latin-1")
        self.assertIn('"Attention"', rows[0]["content"])
        self.assertIn("O(n^2)", rows[0]["content"])

    def test_expand_parent_context_results_dedupes_same_parent(self):
        parent_content = "Parent context with formula and surrounding explanation. " * 20
        results = [
            RetrievalResult(
                id="child-1",
                document="formula child",
                metadata={"parent_id": "parent-1", "parent_content": parent_content},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            ),
            RetrievalResult(
                id="child-2",
                document="nearby child",
                metadata={"parent_id": "parent-1", "parent_content": parent_content},
                score=0.9,
                semantic_score=0.9,
                bm25_score=0.0,
            ),
        ]

        expanded = expand_parent_context_results(results)

        self.assertEqual(len(expanded), 1)
        self.assertIn("surrounding explanation", expanded[0].document)
        self.assertTrue(expanded[0].metadata.get("parent_context_expanded"))
        self.assertNotIn("parent_content", expanded[0].metadata)

    def test_expand_parent_context_results_reads_parent_from_sqlite(self):
        parent_content = "SQLite parent context with formula and surrounding explanation. " * 20
        results = [
            RetrievalResult(
                id="child-1",
                document="formula child",
                metadata={"parent_id": "parent-sqlite"},
                score=1.0,
                semantic_score=1.0,
                bm25_score=0.0,
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            upsert_parent_chunks(
                tmpdir,
                [
                    {
                        "parent_id": "parent-sqlite",
                        "history_key": "test",
                        "url": "https://example.com",
                        "parent_index": 0,
                        "token_count": 120,
                        "content_hash": "hash",
                        "content": parent_content,
                    }
                ],
            )
            expanded = expand_parent_context_results(results, chroma_path=tmpdir)

        self.assertEqual(len(expanded), 1)
        self.assertIn("SQLite parent context", expanded[0].document)
        self.assertEqual(expanded[0].metadata.get("parent_context_store"), "sqlite")

    def test_metadata_signal_score_boosts_matching_queries(self):
        metadata = {
            "chunk_kind": "signal",
            "has_formula_signal": True,
            "has_api_signal": True,
            "has_benchmark_signal": True,
        }

        self.assertGreater(metadata_signal_score("equation formula softmax", metadata), 0)
        self.assertGreater(metadata_signal_score("PyTorch API usage", metadata), 0)
        self.assertGreater(metadata_signal_score("ImageNet benchmark accuracy", metadata), 0)

        table_metadata = {
            "chunk_kind": "table",
            "has_table_signal": True,
            "has_benchmark_signal": True,
        }
        self.assertGreater(metadata_signal_score("benchmark table results", table_metadata), 0)

    def test_source_records_preserve_browser_authority_metadata(self):
        browser_results = [
            {
                "task_id": "task_001",
                "query_context": "Find evidence for topic A.",
                "task_url": "SEARCH:topic A evidence",
                "sources": [
                    {
                        "url": "https://example.edu/source",
                        "title": "Authoritative Source",
                        "source_type": "webpage",
                        "source_quality": "useful_authoritative",
                        "source_authority": "authoritative",
                        "content_noise_score": 0.1,
                        "full_content": "Detailed evidence for topic A. " * 20,
                    }
                ],
            }
        ]

        records = list(source_records(browser_results))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_quality, "useful_authoritative")
        self.assertEqual(records[0].source_authority, "authoritative")
        self.assertEqual(records[0].content_noise_score, 0.1)

    def test_source_records_sanitize_unicode_metadata_and_content(self):
        browser_results = [
            {
                "task_id": "task_\u201c001\u201d",
                "query_context": "Find \u201cquoted\u201d evidence with O(n\u00b2).",
                "task_url": "SEARCH:quoted evidence",
                "sources": [
                    {
                        "url": "https://example.edu/source",
                        "title": "\u201cQuoted Source\u201d",
                        "source_type": "webpage",
                        "source_quality": "primary",
                        "source_authority": "authoritative",
                        "full_content": "\u201cDetailed\u201d evidence with alpha and O(n\u00b2). " * 20,
                    }
                ],
            }
        ]

        record = list(source_records(browser_results))[0]

        self.assertEqual(record.title, '"Quoted Source"')
        self.assertIn('"quoted"', record.query_contexts)
        self.assertIn("O(n^2)", record.query_contexts)
        self.assertIn('"Detailed"', record.content)
        self.assertIn("O(n^2)", record.content)
        for value in (record.title, record.task_ids, record.query_contexts, record.content):
            value.encode("latin-1")

    def test_storage_safe_text_normalizes_smart_quotes_and_math(self):
        text = storage_safe_text("\u201cAttention\u201d has O(n\u00b2) cost and alpha weights.")

        self.assertEqual(text, '"Attention" has O(n^2) cost and alpha weights.')
        text.encode("latin-1")

    def test_upsert_collection_batches_sanitizes_documents_and_metadata(self):
        class RecordingCollection:
            def __init__(self):
                self.documents = []
                self.metadatas = []

            def upsert(self, ids, documents, metadatas):
                self.documents.extend(documents)
                self.metadatas.extend(metadatas)

        collection = RecordingCollection()

        count = upsert_collection_batches(
            collection,
            ["smart"],
            ['Source says \u201cattention\u201d has O(n\u00b2) cost.'],
            [{"title": "\u201cQuoted title\u201d", "symbol": "\u2211 alpha"}],
        )

        self.assertEqual(count, 1)
        self.assertEqual(collection.documents[0], 'Source says "attention" has O(n^2) cost.')
        self.assertEqual(collection.metadatas[0]["title"], '"Quoted title"')
        self.assertEqual(collection.metadatas[0]["symbol"], "sum alpha")
        collection.documents[0].encode("latin-1")
        collection.metadatas[0]["title"].encode("latin-1")

    @patch.dict("os.environ", {"HF_TOKEN": " \u201chf_test_token\u201d "}, clear=True)
    def test_normalize_huggingface_token_env_removes_copied_quotes(self):
        buffer = StringIO()

        with redirect_stdout(buffer):
            normalize_huggingface_token_env()

        self.assertEqual(os.environ["HF_TOKEN"], "hf_test_token")
        self.assertIn("normalized HF_TOKEN", buffer.getvalue())

    @patch.dict("os.environ", {"HF_TOKEN": "hf_\u2603_bad"}, clear=True)
    def test_normalize_huggingface_token_env_rejects_header_unsafe_tokens(self):
        with self.assertRaisesRegex(RuntimeError, "HF_TOKEN contains characters"):
            normalize_huggingface_token_env()

    def test_upsert_collection_batches_logs_non_oom_failures(self):
        class FailingCollection:
            def upsert(self, ids, documents, metadatas):
                raise UnicodeEncodeError("latin-1", "\u201c", 0, 1, "ordinal not in range(256)")

        buffer = StringIO()

        with self.assertRaises(UnicodeEncodeError):
            with redirect_stdout(buffer):
                upsert_collection_batches(
                    FailingCollection(),
                    ["bad"],
                    ["Raw \u201cquote\u201d should be sanitized before logging."],
                    [{"title": "\u201cBad title\u201d"}],
                )

        output = buffer.getvalue()
        self.assertIn("[rag_index] failed during chroma_upsert", output)
        self.assertIn("batch ids=['bad']", output)
        self.assertIn("UnicodeEncodeError", output)

    def test_upsert_collection_batches_retries_cuda_oom_with_smaller_batch(self):
        class OomOnceCollection:
            def __init__(self):
                self.calls = []

            def upsert(self, ids, documents, metadatas):
                self.calls.append(list(ids))
                if len(ids) > 1:
                    raise RuntimeError("CUDA out of memory")

        collection = OomOnceCollection()
        ids = ["a", "b", "c"]
        documents = ["A", "B", "C"]
        metadatas = [{"i": 1}, {"i": 2}, {"i": 3}]

        with patch.dict("os.environ", {"RAG_CHROMA_UPSERT_BATCH_SIZE": "8"}):
            count = upsert_collection_batches(collection, ids, documents, metadatas)

        self.assertEqual(count, 3)
        self.assertEqual(collection.calls, [["a", "b", "c"], ["a"], ["b"], ["c"]])


if __name__ == "__main__":
    unittest.main()
