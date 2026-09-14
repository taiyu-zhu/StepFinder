import os
import json
import argparse
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from Q3Emb import DEFAULT_MODEL_REVISION, Qwen3Embedding


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
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        content_dim: int = 128,
        agent_dim: int = 32,
    ) -> None:
        self.content_dim = content_dim
        self.agent_dim = agent_dim
        self.model = self._load_model(model_path, model_revision)

    def _load_model(
        self, model_path: str, model_revision: str | None
    ) -> Qwen3Embedding:
        revision_text = model_revision or "local/default"
        print(
            f"[INFO] Loading Qwen3-Embedding from {model_path} "
            f"(revision: {revision_text}) ..."
        )
        try:
            model = Qwen3Embedding(
                model_name_or_path=model_path,
                model_revision=model_revision,
            )
            print("[INFO] Model loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load Qwen3-Embedding: {e}")

    def _encode(self, text: str, dim: int) -> np.ndarray:
        """
        Encode a single text string into a fixed-dimension embedding.

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
            raise RuntimeError(f"Failed to encode text: {e}") from e

    @staticmethod
    def _mistake_labels(
        log: dict[str, Any], history: list[dict[str, Any]]
    ) -> np.ndarray:
        """Build one-hot step labels from either supported Who&When format."""
        has_step_labels = ["is_mistake" in step for step in history]
        root_mistake_step = log.get("mistake_step")
        mistake_idx = None

        if root_mistake_step is not None:
            try:
                mistake_idx = int(root_mistake_step)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid mistake_step value: {root_mistake_step!r}"
                ) from e
            if not 0 <= mistake_idx < len(history):
                raise ValueError(
                    f"mistake_step {mistake_idx} is outside history length "
                    f"{len(history)}"
                )

        if all(has_step_labels):
            labels = np.array(
                [int(step["is_mistake"]) for step in history], dtype=np.int64
            )
        elif mistake_idx is not None:
            labels = np.zeros(len(history), dtype=np.int64)
            labels[mistake_idx] = 1
        else:
            raise ValueError(
                "Missing labels: provide history[*].is_mistake or root mistake_step"
            )

        if not np.isin(labels, [0, 1]).all() or int(labels.sum()) != 1:
            raise ValueError(
                "Expected exactly one mistake step, but labels are "
                f"{labels.tolist()}"
            )

        if mistake_idx is not None and all(has_step_labels):
            if int(labels.argmax()) != mistake_idx:
                raise ValueError(
                    "history[*].is_mistake disagrees with root mistake_step: "
                    f"{int(labels.argmax())} != {mistake_idx}"
                )

        return labels

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
            raise ValueError("history is empty")

        content_features = []
        agent_features = []
        mistake_labels = self._mistake_labels(log, history)

        for step in history:
            content = step.get("content", "")
            agent = (step.get("name") or step.get("role") or "").strip()

            content_features.append(self._encode(content, self.content_dim))
            agent_features.append(self._encode(agent, self.agent_dim))

        return {
            "content_features": np.stack(content_features, axis=0),
            "agent_features": np.stack(agent_features, axis=0),
            "mistake_labels": mistake_labels,
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
    overwrite_cache: bool = False,
) -> list:
    """
    Encode all JSON execution logs in data_dir into embedding features.
    Already-cached files are skipped automatically.

    Returns:
        List of feature dicts for all files (cached + newly encoded).
    """
    all_features = []
    failures = []
    file_list = (
        filenames
        if filenames is not None
        else sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    )

    for filename in tqdm(file_list, desc=f"Encoding {os.path.basename(data_dir)}"):
        cache_path = _get_cache_path(data_dir, filename)

        if os.path.exists(cache_path) and not overwrite_cache:
            all_features.append(torch.load(cache_path, weights_only=False))
            continue

        json_path = os.path.join(data_dir, filename)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[WARNING] Failed to read {filename}: {e}")
            failures.append(filename)
            continue

        try:
            features = extractor.process_log(log_data)
            torch.save(features, cache_path)
            all_features.append(features)
        except Exception as e:
            print(f"[WARNING] Failed to encode {filename}: {e}")
            failures.append(filename)

    if failures:
        preview = ", ".join(failures[:5])
        if len(failures) > 5:
            preview += f", ... ({len(failures)} files total)"
        raise RuntimeError(f"Feature construction failed for: {preview}")

    return all_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode execution logs into temporal semantic sequences."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON log files.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Path to the Qwen3-Embedding model.")
    parser.add_argument(
        "--model_revision",
        type=str,
        default=DEFAULT_MODEL_REVISION,
        help="Exact Hugging Face model revision. Pass an empty string for a local model without a revision.",
    )
    parser.add_argument("--content_dim", type=int, default=128, help="Content embedding dimension.")
    parser.add_argument("--agent_dim", type=int, default=32, help="Agent embedding dimension.")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Regenerate existing cached features.",
    )
    args = parser.parse_args()

    extractor = TemporalFeatureExtractor(
        model_path=args.model_path,
        model_revision=args.model_revision or None,
        content_dim=args.content_dim,
        agent_dim=args.agent_dim,
    )
    encode_dir(args.data_dir, extractor, overwrite_cache=args.overwrite_cache)
