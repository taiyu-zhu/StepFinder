import torch
from typing import List, Dict, Any


def sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate a list of feature dicts into a padded batch.

    Each sample in the batch is expected to contain:
        content_features: np.ndarray [T, content_dim]
        agent_features:   np.ndarray [T, agent_dim]
        mistake_labels:   np.ndarray [T], int64

    Returns:
        Dict with keys:
            content_seq:    [B, T_max, content_dim]
            agent_seq:      [B, T_max, agent_dim]
            mistake_labels: [B, T_max]
            mask:           [B, T_max]  1 = valid step, 0 = padding
    """
    batch_size = len(batch)
    T_max = max(sample["content_features"].shape[0] for sample in batch)
    D_content = batch[0]["content_features"].shape[1]
    D_agent = batch[0]["agent_features"].shape[1]

    content_seq = torch.zeros((batch_size, T_max, D_content), dtype=torch.float32)
    agent_seq = torch.zeros((batch_size, T_max, D_agent), dtype=torch.float32)
    mistake_labels = torch.zeros((batch_size, T_max), dtype=torch.long)
    mask = torch.zeros((batch_size, T_max), dtype=torch.float32)

    for i, sample in enumerate(batch):
        L = sample["content_features"].shape[0]
        content_seq[i, :L] = torch.tensor(sample["content_features"], dtype=torch.float32)
        agent_seq[i, :L] = torch.tensor(sample["agent_features"], dtype=torch.float32)
        mistake_labels[i, :L] = torch.tensor(sample["mistake_labels"], dtype=torch.long)
        mask[i, :L] = 1.0

    return {
        "content_seq": content_seq,
        "agent_seq": agent_seq,
        "mistake_labels": mistake_labels,
        "mask": mask,
    }