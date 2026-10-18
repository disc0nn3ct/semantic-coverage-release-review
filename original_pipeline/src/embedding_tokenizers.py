"""Pinned tokenizer-only preparation for MX/Nomic embedding endpoint inputs.

No model weights are loaded, and this module makes no network/API calls.
Counts are exact for the pinned HF tokenizer, not automatically for a GGUF backend.
"""
from pathlib import Path
import hashlib
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "tokenizer_manifest.json"
PREFIXES = {
    "mx": {"code": "Represent this sentence for searching relevant passages: ", "document": ""},
    "nomic": {"code": "search_query: ", "document": "search_document: "},
}
DIMENSIONS = {"mx": 1024, "nomic": 768}

class EmbeddingTokenizer:
    def __init__(self, name, manifest_path=MANIFEST):
        if name not in PREFIXES:
            raise ValueError("name must be mx or nomic")
        self.name = name
        self.manifest_path = Path(manifest_path).resolve()
        record = json.loads(self.manifest_path.read_text(encoding="utf-8"))["tokenizers"][name]
        self.path = self.manifest_path.parent.parent / record["path"]
        for filename, expected in record["files"].items():
            data = (self.path / filename).read_bytes()
            if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"Tokenizer artifact mismatch: {filename}")
        self.record = record
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False, use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError("A fast tokenizer with offsets is required to preserve input characters")
        self.tokenizer.model_max_length = 2**31 - 1

    def count(self, text, special_tokens=True):
        return len(self.tokenizer(text, add_special_tokens=special_tokens, truncation=False, return_attention_mask=False)["input_ids"])

    def chunks(self, text, role):
        """Preserve all source characters; each payload is prefix + raw substring.

        At most 384 re-encoded content tokens/chunk and 512 tokens/payload with
        prefix and special tokens. Empty/whitespace-only inputs yield no requests.
        A wordpiece boundary can change tokenization, so every slice is recounted.
        """
        if not isinstance(text, str) or role not in {"code", "document"}:
            raise ValueError("Provide a text string and role code/document")
        prefix = PREFIXES[self.name][role]
        encoded = self.tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True, return_attention_mask=False)
        offsets = encoded["offset_mapping"]
        if not offsets:
            return []
        chunks, start_token, start_char = [], 0, 0
        while start_token < len(offsets):
            end_token = min(start_token + 384, len(offsets))
            while True:
                end_char = offsets[end_token][0] if end_token < len(offsets) else len(text)
                raw = text[start_char:end_char]
                content_tokens = self.count(raw, special_tokens=False)
                payload = prefix + raw
                payload_tokens = self.count(payload, special_tokens=True)
                if end_char > start_char and 0 < content_tokens <= 384 and payload_tokens <= 512:
                    break
                end_token -= 1
                if end_token <= start_token:
                    raise ValueError("Cannot construct a safe nonempty chunk without losing source text")
            chunks.append({"index": len(chunks), "role": role, "text": payload, "raw_text": raw,
                           "char_start": start_char, "char_end": end_char,
                           "content_tokens": content_tokens, "payload_tokens_with_specials": payload_tokens,
                           "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()})
            start_token, start_char = end_token, end_char
        assert "".join(chunk["raw_text"] for chunk in chunks) == text
        return chunks

    def provenance(self):
        return {"repo_id": self.record["repo_id"], "revision": self.record["revision"],
                "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tokenizer_class": type(self.tokenizer).__name__, "content_limit": 384, "payload_limit": 512,
                "prefixes": PREFIXES[self.name], "expected_dimension": DIMENSIONS[self.name],
                "backend_tokenizer_equivalence": "not established by this tokenizer-only check"}

def pool_chunk_vectors(vectors, chunks, model_name):
    """Normalize each full-dimensional vector, token-weight mean, normalize again."""
    if model_name not in DIMENSIONS or len(vectors) != len(chunks):
        raise ValueError("Unknown model or vector/chunk count mismatch")
    dimension = DIMENSIONS[model_name]
    if not chunks:
        return np.zeros(dimension, dtype=np.float32)
    values = np.asarray(vectors, dtype=np.float64)
    if values.shape != (len(chunks), dimension) or not np.isfinite(values).all():
        raise ValueError("Invalid embedding shape/values; dimension truncation is forbidden")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths == 0):
        raise ValueError("Nonempty chunk produced a zero vector")
    weights = np.asarray([chunk["content_tokens"] for chunk in chunks], dtype=np.float64)
    if np.any(weights <= 0) or np.any(weights > 384):
        raise ValueError("Invalid content-token weight")
    pooled = np.sum((values / lengths[:, None]) * weights[:, None], axis=0) / weights.sum()
    norm = np.linalg.norm(pooled)
    if norm == 0:
        raise ValueError("Chunk embeddings cancel to an undefined direction")
    return (pooled / norm).astype(np.float32)
