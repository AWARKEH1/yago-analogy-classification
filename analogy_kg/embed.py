"""Compute one sentence embedding per entity with bge-large-en-v1.5.

The original project embedded every entity of every analogy row and stored the
4 x 1024 floats as text inside a CSV. Here each *unique* entity is embedded once
and stored in a compact ``.npz`` file (entity -> vector), which is much faster
and lighter.

Usage:
    python -m analogy_kg.embed --analogies data/analogies --out data/embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import entity_to_text

# The original project used Alibaba-NLP/gte-large-en-v1.5, whose custom
# (trust_remote_code) implementation crashes with recent versions of
# transformers. BAAI/bge-large-en-v1.5 is a standard BERT model with the same
# embedding size (1024), so the ANNc architecture is unchanged.
MODEL_NAME = "BAAI/bge-large-en-v1.5"


class Embedder:
    """Mean-pooled sentence embeddings (same method as the original project)."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i:i + batch_size], padding=True,
                                   truncation=True, max_length=64,
                                   return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"]).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analogies", type=Path, default=Path("data/analogies"))
    parser.add_argument("--out", type=Path, default=Path("data/embeddings.npz"))
    args = parser.parse_args()

    entities: set[str] = set()
    for csv in sorted(args.analogies.glob("*.csv")):
        df = pd.read_csv(csv, dtype=str, keep_default_na=False)
        for col in "abcd":
            entities.update(df[col].astype(str))
    entities_list = sorted(entities)
    print(f"Embedding {len(entities_list)} unique entities with {MODEL_NAME}...")

    vectors = Embedder().encode([entity_to_text(e) for e in entities_list])
    np.savez_compressed(args.out, entities=np.array(entities_list), vectors=vectors)
    print(f"Saved {vectors.shape} to {args.out}")


if __name__ == "__main__":
    main()
