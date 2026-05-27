import os
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import StepFinder, compute_loss
from collate_fn import sequence_collate_fn


# ---------------------------------------------------------------------------
# Set seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cache_path(data_dir: str, filename: str) -> str:
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, os.path.splitext(filename)[0] + "_multi_features.pt")


def load_data_from_dir(data_dir: str) -> list:
    all_seqs = []
    if not os.path.exists(data_dir):
        print(f"[WARNING] Directory not found: {data_dir}")
        return all_seqs

    file_list = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    for filename in tqdm(file_list, desc=f"Loading {os.path.basename(data_dir)}"):
        cpath = get_cache_path(data_dir, filename)
        if os.path.exists(cpath):
            all_seqs.append(torch.load(cpath, weights_only=False))
        else:
            print(f"[WARNING] Cache not found, skipping: {cpath}")
    return all_seqs


class SequenceDataset(Dataset):
    def __init__(self, data: list) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: StepFinder, dataloader: DataLoader, device: torch.device) -> dict:
    """
    Evaluate step-level failure attribution performance.

    Metrics (following Who&When benchmark protocol):
        acc:           Accuracy — Top-1 prediction matches ground truth.
        top2:          Acc@2    — ground truth appears in Top-2 predictions.
        top3:          Acc@3    — ground truth appears in Top-3 predictions.
        mrr:           MRR@3   — mean reciprocal rank, truncated at K=3.
        tolerance_acc: Tolerance Accuracy — Top-1 prediction within δ steps of ground truth,
                       for δ in {1, 2, 3, 4, 5}.

    Returns:
        dict with keys: acc, top2, top3, mrr, tolerance_acc.
    """
    model.eval()

    hits = 0
    top2_hits = 0
    top3_hits = 0
    total = 0
    all_inv_ranks = []
    tolerance_hits = {delta: 0 for delta in range(1, 6)}

    with torch.no_grad():
        for batch in dataloader:
            mask = batch["mask"].to(device)
            content = batch["content_seq"].to(device)
            agent = batch["agent_seq"].to(device)
            targets = batch["mistake_labels"].to(device)

            logits, _ = model(content, agent, mask)
            logits = logits.masked_fill(mask == 0, float("-inf"))

            _, top_indices = torch.sort(logits, dim=1, descending=True)

            for i in range(len(targets)):
                total += 1
                true_idx = int(torch.argmax(targets[i]).item())
                top1_idx = int(top_indices[i, 0].item())

                if top1_idx == true_idx:
                    hits += 1
                if true_idx in top_indices[i, :2]:
                    top2_hits += 1
                if true_idx in top_indices[i, :3]:
                    top3_hits += 1

                # MRR@3
                rank_tensor = (top_indices[i] == true_idx).nonzero(as_tuple=True)[0]
                rank = int(rank_tensor[0].item()) + 1
                all_inv_ranks.append(1.0 / rank if rank <= 3 else 0.0)

                # Tolerance Accuracy
                for delta in range(1, 6):
                    if abs(top1_idx - true_idx) <= delta:
                        tolerance_hits[delta] += 1

    return {
        "acc": hits / total if total > 0 else 0.0,
        "top2": top2_hits / total if total > 0 else 0.0,
        "top3": top3_hits / total if total > 0 else 0.0,
        "mrr": float(np.mean(all_inv_ranks)) if all_inv_ranks else 0.0,
        "tolerance_acc": {delta: tolerance_hits[delta] / total for delta in range(1, 6)}
        if total > 0 else {delta: 0.0 for delta in range(1, 6)},
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Running on {device}")

    # --- Data ---
    train_data = load_data_from_dir(args.train_dir)
    test_data = load_data_from_dir(args.test_dir)

    if not train_data:
        print("[ERROR] Training data is empty. Aborting.")
        return

    train_loader = DataLoader(
        SequenceDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=sequence_collate_fn,
    )
    test_loader = DataLoader(
        SequenceDataset(test_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
    )

    # --- Model ---
    model = StepFinder(
        content_dim=args.content_dim,
        agent_dim=args.agent_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        scales=args.scales,
        dropout=args.dropout,
    ).to(device)

    # --- Create output directory if needed ---
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_acc = 0.0
    stagnant_epochs = 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            mask = batch["mask"].to(device)
            content = batch["content_seq"].to(device)
            agent = batch["agent_seq"].to(device)
            targets = batch["mistake_labels"].to(device)

            logits, temporal_loss = model(content, agent, mask)
            loss = compute_loss(
                logits, targets, mask, temporal_loss,
                lambda_temporal=args.lambda_temporal,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] avg loss: {avg_loss:.4f}")

        # --- Evaluate ---
        res = evaluate(model, test_loader, device)
        print(
            f"[Epoch {epoch + 1}] "
            f"Acc: {res['acc']:.4f} | "
            f"Acc@2: {res['top2']:.4f} | "
            f"Acc@3: {res['top3']:.4f} | "
            f"MRR@3: {res['mrr']:.4f}"
        )
        tol = res["tolerance_acc"]
        print(
            f"           Tolerance Acc: "
            f"δ1={tol[1]:.4f} | δ2={tol[2]:.4f} | δ3={tol[3]:.4f} | "
            f"δ4={tol[4]:.4f} | δ5={tol[5]:.4f}"
        )

        # --- Early stopping & checkpointing ---
        if res["acc"] > best_acc:
            best_acc = res["acc"]
            stagnant_epochs = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"[INFO] New best model saved (Acc: {best_acc:.4f}) -> {args.save_path}")
        else:
            stagnant_epochs += 1
            print(f"[INFO] No improvement. Stagnant: {stagnant_epochs}/{args.patience}")

        if stagnant_epochs >= args.patience:
            print(f"[INFO] Early stopping triggered after {args.patience} stagnant epochs.")
            break

    print(f"[INFO] Done. Best Acc: {best_acc:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train StepFinder")

    # Paths
    parser.add_argument("--train_dir", type=str, default="Algorithm_train")
    parser.add_argument("--test_dir", type=str, default="Algorithm_test")
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--device", type=str, default="cuda")

    # Model
    parser.add_argument("--content_dim", type=int, default=128)
    parser.add_argument("--agent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--dropout", type=float, default=0.5)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_temporal", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
