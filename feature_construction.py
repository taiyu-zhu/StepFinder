import os
import json
import argparse
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from Q3Emb import Qwen3Embedding


class TemporalFeatureExtractor:
    """
    Converts a multi-agent execution log into a temporal semantic sequence.

    Each step in the log is encoded into:
        - A content embedding (c_t) capturing the action content.
        - An agent embedding  (r_t) capturing the executing agent identity.
    """

    def __init__(
        self,
        model_path: str,
        content_dim: int = 128,
        agent_dim: int = 32,
    ) -> None:
        self.content_dim = content_dim
        self.agent_dim = agent_dim
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> Qwen3Embedding:
        print(f"[INFO] Loading Qwen3-Embedding from {model_path} ...")
        try:
            model = Qwen3Embedding(model_name_or_path=model_path)
            print("[INFO] Model loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load Qwen3-Embedding: {e}")

    def _encode(self, text: str, dim: int) -> np.ndarray:
        """
        Encode a single text string into a fixed-dimension embedding.

        Args:
            text: Input text to encode.
            dim:  Target embedding dimension.

        Returns:
            Embedding array of shape [dim], dtype float32.
        """
        if not text or not text.strip():
            return np.zeros(dim, dtype=np.float32)

        try:
            tensor = self.model.encode(text, dim=dim)
            embedding = tensor.squeeze(0).cpu().numpy().astype(np.float32)

            if embedding.shape[0] > dim:
                embedding = embedding[:dim]
            elif embedding.shape[0] < dim:
                embedding = np.pad(
                    embedding, (0, dim - embedding.shape[0]), mode="constant"
                )
            return embedding
        except Exception as e:
            print(f"[WARNING] Encoding failed ({e}), returning zero vector.")
            return np.zeros(dim, dtype=np.float32)

    def process_log(self, log: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single execution log into temporal embedding features.

        Returns:
            Dict with keys:
                content_features: np.ndarray [T, content_dim]
                agent_features:   np.ndarray [T, agent_dim]
                mistake_labels:   np.ndarray [T], int64
        """
        history: list[dict[str, Any]] = log.get("history", [])

        if not history:
            return {
                "content_features": np.zeros((0, self.content_dim), dtype=np.float32),
                "agent_features": np.zeros((0, self.agent_dim), dtype=np.float32),
                "mistake_labels": np.zeros(0, dtype=np.int64),
            }

        content_features = []
        agent_features = []
        mistake_labels = []

        for step in history:
            content = step.get("content", "")
            agent = (step.get("name") or step.get("role") or "").strip()

            content_features.append(self._encode(content, self.content_dim))
            agent_features.append(self._encode(agent, self.agent_dim))
            mistake_labels.append(int(step.get("is_mistake", 0)))

        return {
            "content_features": np.stack(content_features, axis=0),
            "agent_features": np.stack(agent_features, axis=0),
            "mistake_labels": np.array(mistake_labels, dtype=np.int64),
        }


def _get_cache_path(data_dir: str, filename: str) -> str:
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.splitext(filename)[0]
    return os.path.join(cache_dir, f"{base_name}_multi_features.pt")


def encode_dir(
    data_dir: str,
    extractor: TemporalFeatureExtractor,
    filenames: list[str] | None = None,
) -> list:
    """
    Encode all JSON execution logs in data_dir into embedding features.
    Already-cached files are skipped automatically.

    Returns:
        List of feature dicts for all files (cached + newly encoded).
    """
    all_features = []
    file_list = (
        filenames
        if filenames is not None
        else [f for f in os.listdir(data_dir) if f.endswith(".json")]
    )

    for filename in tqdm(file_list, desc=f"Encoding {os.path.basename(data_dir)}"):
        cache_path = _get_cache_path(data_dir, filename)

        if os.path.exists(cache_path):
            all_features.append(torch.load(cache_path, weights_only=False))
            continue

        json_path = os.path.join(data_dir, filename)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARNING] Failed to read {filename}: {e}")
            continue

        try:
            features = extractor.process_log(log_data)
            torch.save(features, cache_path)
            all_features.append(features)
        except Exception as e:
            print(f"[WARNING] Failed to encode {filename}: {e}")

    return all_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode execution logs into temporal semantic sequences."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON log files.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Path to the Qwen3-Embedding model.")
    parser.add_argument("--content_dim", type=int, default=128, help="Content embedding dimension.")
    parser.add_argument("--agent_dim", type=int, default=32, help="Agent embedding dimension.")
    args = parser.parse_args()

    extractor = TemporalFeatureExtractor(
        model_path=args.model_path,
        content_dim=args.content_dim,
        agent_dim=args.agent_dim,
    )
    encode_dir(args.data_dir, extractor)