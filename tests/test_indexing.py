import unittest

from src.rag.indexing import langchain_ingestion_classes, split_document
from src.rag.retrieval import metadata_signal_score


class IndexingChunkingTests(unittest.TestCase):
    def test_split_document_adds_signal_chunks_for_precise_evidence(self):
        Document, _ = langchain_ingestion_classes()
        document = Document(
            page_content="""Attention paper notes.

The formula is Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V.

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

    def test_metadata_signal_score_boosts_matching_queries(self):
        metadata = {
            "chunk_kind": "signal",
            "has_formula_signal": True,
            "has_api_signal": True,
            "has_benchmark_signal": True,
        }

        self.assertGreater(metadata_signal_score("scaled dot-product attention equation softmax", metadata), 0)
        self.assertGreater(metadata_signal_score("PyTorch API usage", metadata), 0)
        self.assertGreater(metadata_signal_score("ImageNet benchmark accuracy", metadata), 0)


if __name__ == "__main__":
    unittest.main()
