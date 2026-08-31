import json
import tempfile
import unittest

from src.rag.indexing import (
    clean_document_text,
    langchain_ingestion_classes,
    parent_content_for_id,
    parent_context_chunks,
    parent_rows_from_documents,
    source_records,
    split_document,
    structure_aware_chunks,
    strip_parent_content_metadata,
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


if __name__ == "__main__":
    unittest.main()
