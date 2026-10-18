"""Pinned CPU-only CodeBERT/GraphCodeBERT sequence encoders; no fitting or labels.

GraphCodeBERT here uses its real pretrained weights on token sequences without
data-flow nodes. It is NOT the paper's fine-tuned graph-aware retrieval method.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "models" / "frozen_model_manifest.json"
PROTOCOL = {
    "name": "frozen-sequence-mean-v1",
    "pooling": "final-layer content-token mean, excluding BOS/EOS/padding; token-count weighted across all nonoverlapping chunks; L2 normalize",
    "chunk_content_tokens": 510,
    "chunk_overlap": 0,
    "max_chunks": None,
    "empty_text": "zero 768-vector; cosine involving zero is zero",
    "input": "verbatim caller-supplied text; no automatic prefix, summary, identifiers, labels, or target paths added",
    "fine_tuning": False,
    "graph_data_flow": False,
    "device": "cpu",
    "dtype": "float32",
}

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

class FrozenEncoder:
    def __init__(self, model_name, manifest_path=DEFAULT_MANIFEST, threads=4, batch_size=8):
        if model_name not in {"codebert", "graphcodebert"}:
            raise ValueError("model_name must be codebert or graphcodebert")
        if threads < 1 or batch_size < 1:
            raise ValueError("threads and batch_size must be positive")
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entry = manifest["models"][model_name]
        # Paths in this manifest are relative to the study root, one above models/.
        self.model_path = self.manifest_path.parent.parent / self.entry["path"]
        for name, evidence in self.entry["files"].items():
            path = self.model_path / name
            if not path.is_file() or path.stat().st_size != evidence["size_bytes"] or sha256_file(path) != evidence["sha256"]:
                raise ValueError(f"Frozen artifact mismatch: {name}")
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False, use_fast=True)
        self.tokenizer.model_max_length = 2**31 - 1  # We chunk explicitly, never truncate silently.
        self.model = AutoModel.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False, add_pooling_layer=False, torch_dtype=torch.float32)
        self.model.to(torch.device("cpu"))
        self.model.eval()
        self.model.requires_grad_(False)
        assert all(p.device.type == "cpu" and not p.requires_grad for p in self.model.parameters())
        assert self.model.config.hidden_size == 768
        self.model_name, self.batch_size, self.threads = model_name, batch_size, threads
        self.last_metadata = []

    def encode(self, texts):
        if isinstance(texts, str):
            raise TypeError("encode expects a list of text strings")
        texts = list(texts)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("encode expects a list of text strings")
        token_lists = [self.tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"] for text in texts]
        work = [(index, ids[offset:offset + 510]) for index, ids in enumerate(token_lists) for offset in range(0, len(ids), 510)]
        sums = np.zeros((len(texts), 768), dtype=np.float64)
        counts = np.zeros(len(texts), dtype=np.int64)
        bos, eos, pad = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        with torch.inference_mode():
            for start in range(0, len(work), self.batch_size):
                batch = work[start:start + self.batch_size]
                longest = max(len(tokens) + 2 for _, tokens in batch)
                input_ids = torch.full((len(batch), longest), pad, dtype=torch.long, device="cpu")
                attention = torch.zeros_like(input_ids)
                content_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for row, (_, tokens) in enumerate(batch):
                    size = len(tokens) + 2
                    input_ids[row, :size] = torch.tensor([bos] + tokens + [eos], dtype=torch.long)
                    attention[row, :size] = 1
                    content_mask[row, 1:size - 1] = True
                hidden = self.model(input_ids=input_ids, attention_mask=attention).last_hidden_state
                totals = (hidden.double() * content_mask.unsqueeze(-1)).sum(dim=1).cpu().numpy()
                for row, (index, tokens) in enumerate(batch):
                    sums[index] += totals[row]
                    counts[index] += len(tokens)
        vectors = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] != 0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0).astype(np.float32)
        assert np.isfinite(vectors).all()
        self.last_metadata = [{"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "characters": len(text), "content_tokens": len(ids), "chunks": (len(ids) + 509) // 510, "truncated_tokens": 0} for text, ids in zip(texts, token_lists)]
        return vectors

    def similarity_matrix(self, requirement_texts, change_texts):
        """Return raw cosine in [-1,1]; rows=requirements, columns=changes."""
        requirement_vectors = self.encode(requirement_texts)
        change_vectors = self.encode(change_texts)
        return np.clip(requirement_vectors @ change_vectors.T, -1.0, 1.0)

    def provenance(self):
        return {
            "model": self.model_name, "repo_id": self.entry["repo_id"], "revision": self.entry["revision"],
            "model_manifest_sha256": sha256_file(self.manifest_path), "encoder_source_sha256": sha256_file(__file__),
            "protocol": PROTOCOL, "threads": self.threads, "batch_size": self.batch_size,
            "runtime": {"python": platform.python_version(), "platform": platform.platform(), **{name: importlib.metadata.version(name) for name in ["torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "numpy"]}},
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["codebert", "graphcodebert"], required=True)
    parser.add_argument("--input", type=Path, required=True, help='JSON array containing exactly {"id": string, "text": string} objects')
    parser.add_argument("--output", type=Path, required=True, help="output .npz path; metadata saved beside it")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(set(row) != {"id", "text"} or not isinstance(row["id"], str) or not isinstance(row["text"], str) for row in rows):
        raise ValueError("Only id/text fields are accepted; evaluation labels must remain separate")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Input IDs must be unique")
    started = time.perf_counter()
    encoder = FrozenEncoder(args.model, args.manifest, args.threads, args.batch_size)
    vectors = encoder.encode([row["text"] for row in rows])
    elapsed = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.asarray([row["id"] for row in rows]), embeddings=vectors)
    metadata = {**encoder.provenance(), "input_sha256": sha256_file(args.input), "elapsed_seconds_including_load_and_validation": elapsed, "items": [{"id": row["id"], **meta} for row, meta in zip(rows, encoder.last_metadata)]}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "shape": list(vectors.shape), "elapsed_seconds": elapsed}))

if __name__ == "__main__":
    main()
