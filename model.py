import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFeatureExtraction(nn.Module):
    """
    Temporal Feature Extraction (TFE) module.

    Encodes content embeddings into bidirectional hidden representations
    via LayerNorm -> BiLSTM -> LayerNorm.
    """

    def __init__(
        self,
        content_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(content_dim)
        self.bilstm = nn.LSTM(
            input_size=content_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.output_ln = nn.LayerNorm(hidden_dim * 2)

    def forward(
        self,
        content_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            H_out: Normalized BiLSTM hidden states [B, T, 2 * hidden_dim].
        """
        B, T, _ = content_seq.shape
        mask_f = mask.unsqueeze(-1).float()

        x = self.input_ln(content_seq) * mask_f

        lengths = mask.sum(dim=1).cpu().long().clamp(min=1)
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.bilstm(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=T
        )

        out = out * mask_f
        out = self.output_ln(out) * mask_f
        return out


class AgentAwareStepInteraction(nn.Module):
    """
    Agent-Aware Step Interaction (ASI) module.

    Each trajectory step attends to every other step via scaled dot-product
    attention (Agent-Aware Bias), followed by a global agent gating mechanism
    (Agent-Aware Gating) that modulates step features based on the collective
    agent profile of the trajectory.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        agent_dim: int = 32,
        num_heads: int = 2,
        head_dim: int = 32,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.alpha = alpha

        # Linear projections for multi-head attention
        self.W_Q = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.W_K = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.W_V = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.out_proj = nn.Linear(head_dim * num_heads, hidden_dim)

        # Agent-Aware Gating: maps mean agent embedding to per-feature gates
        self.gating = nn.Linear(agent_dim, hidden_dim)

    def forward(
        self,
        H_out: torch.Tensor,
        agent_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            H_final: Gated, attention-augmented step features [B, T, hidden_dim].
        """
        B, T, H = H_out.shape
        mask_f = mask.unsqueeze(-1).float()

        # --- Multi-head projections ---
        Q = self.W_Q(H_out).view(B, T, self.num_heads, self.head_dim)
        K = self.W_K(H_out).view(B, T, self.num_heads, self.head_dim)
        V = self.W_V(H_out).view(B, T, self.num_heads, self.head_dim)

        # Scaled dot-product interaction matrix E  [B, num_heads, T, T]
        E = torch.einsum("bthd,bshd->bhts", Q, K) / (self.head_dim ** 0.5)

        # --- Agent-Aware Bias  B_ij = cos(r_i, r_j) ---
        agent_norm = F.normalize(agent_seq, p=2, dim=-1)
        B_mat = torch.bmm(agent_norm, agent_norm.transpose(1, 2))  # [B, T, T]

        # Attn = softmax(E + alpha * B)
        attn_scores = E + self.alpha * B_mat.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(
            mask.unsqueeze(1).unsqueeze(2) == 0, -1e9
        )
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Aggregate and project back  ->  H_attn = H_out + Attn * V
        attn_out = torch.einsum("bhts,bshd->bthd", attn_probs, V)
        attn_out = attn_out.reshape(B, T, self.num_heads * self.head_dim)
        H_attn = H_out + self.out_proj(attn_out)
        H_attn = H_attn * mask_f

        # --- Agent-Aware Gating ---
        # Global mean agent embedding (masked average)
        r_bar = (agent_seq * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-6)

        # g = sigmoid(W_g * r_bar + b_g)
        g = torch.sigmoid(self.gating(r_bar))  # [B, hidden_dim]

        # H_final = H_attn ⊙ (1 + alpha * g)
        H_final = H_attn * (1.0 + self.alpha * g.unsqueeze(1))
        H_final = H_final * mask_f
        return H_final


class StepLevelErrorScoring(nn.Module):
    """
    Step-Level Error Scoring module.

    Produces a scalar anomaly score for each trajectory step by combining:
      1. A preliminary error score from a two-layer MLP.
      2. A multi-scale temporal differencing signal (beta * Delta_t).
      3. A linearly decaying position bias that discourages late attributions
         (gamma * b_t).

    Also exposes a lightweight temporal consistency head used during training
    to predict the next step's hidden representation (self-supervised auxiliary task).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        scales: list[int] = None,
        beta: float = 0.9,
        gamma: float = 0.4,
    ) -> None:
        super().__init__()
        self.scales = scales if scales is not None else [1, 2]
        self.beta = beta
        self.gamma = gamma

        # Preliminary error scoring MLP: H_final -> scalar
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Temporal consistency head: f_theta(h_t) -> h_{t+1}
        self.temporal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _compute_multi_scale_diff(
        self,
        H: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute normalized multi-scale temporal difference scores Delta_t.

        Returns:
            Delta: Averaged normalized difference scores [B, T].
        """
        B, T, _ = H.shape
        device = H.device
        all_diffs = []

        for s in self.scales:
            if s == 1:
                delta = H[:, 1:] - H[:, :-1]                              # [B, T-1, D]
            else:
                delta = (
                    H[:, s:]
                    - s * H[:, 1 : T - s + 1]
                    + (s - 1) * H[:, : T - s]
                )                                                          # [B, T-s, D]

            delta_norm = delta.norm(dim=-1)                                # [B, T-s or T-1]
            delta_norm = torch.cat(
                [torch.zeros(B, s, device=device), delta_norm], dim=1
            )                                                              # [B, T]

            # Normalize by trajectory mean (per sample)
            mean_val = (delta_norm * mask.float()).sum(dim=1, keepdim=True) / (
                mask.float().sum(dim=1, keepdim=True) + 1e-6
            )
            delta_norm = delta_norm / (mean_val + 1e-6)
            all_diffs.append(delta_norm)

        return sum(all_diffs) / len(all_diffs)                            # [B, T]

    def forward(
        self,
        H_final: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:          Final step-level error scores [B, T].
            temporal_loss:   Unweighted temporal consistency MSE loss (scalar).
        """
        B, T, _ = H_final.shape

        # --- Preliminary error scores (Eq. 6) ---
        logits = self.mlp(H_final).squeeze(-1)                            # [B, T]

        # --- Multi-scale differencing (Eq. 7-8) ---
        delta = self._compute_multi_scale_diff(H_final, mask)             # [B, T]
        logits = logits + self.beta * delta

        # --- Position bias b_t = -(t-1)/(T-1)  (Eq. 9) ---
        t_idx = torch.arange(T, device=logits.device).float()
        b_t = -(t_idx / (T - 1 + 1e-6))                                  # [T]
        logits = logits + self.gamma * b_t.unsqueeze(0)

        # --- Temporal consistency head (Eq. 12) ---
        pred = self.temporal_head(H_final[:, :-1])                        # [B, T-1, D]
        target = H_final[:, 1:].detach()
        pred_mask = mask[:, 1:].unsqueeze(-1).float()
        temporal_loss = F.mse_loss(
            pred * pred_mask,
            target * pred_mask,
            reduction="sum",
        ) / (pred_mask.sum() + 1e-6)

        return logits, temporal_loss


class StepFinder(nn.Module):
    """
    Pipeline:
        content_seq, agent_seq
            -> TemporalFeatureExtraction
            -> AgentAwareStepInteraction
            -> StepLevelErrorScoring
            -> logits [B, T],  temporal_loss
    """

    def __init__(
        self,
        content_dim: int = 128,
        agent_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 2,
        head_dim: int = 32,
        alpha: float = 0.1,
        beta: float = 0.9,
        gamma: float = 0.4,
        scales: list[int] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.tfe = TemporalFeatureExtraction(
            content_dim=content_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.asi = AgentAwareStepInteraction(
            hidden_dim=hidden_dim * 2,
            agent_dim=agent_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            alpha=alpha,
        )
        self.scoring = StepLevelErrorScoring(
            hidden_dim=hidden_dim * 2,
            scales=scales if scales is not None else [1, 2],
            beta=beta,
            gamma=gamma,
        )

    def forward(
        self,
        content_seq: torch.Tensor,
        agent_seq: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits:        Step-level error scores [B, T].
            temporal_loss: Unweighted temporal consistency loss (scalar).
        """
        H_out = self.tfe(content_seq, mask)
        H_final = self.asi(H_out, agent_seq, mask)
        logits, temporal_loss = self.scoring(H_final, mask)
        return logits, temporal_loss


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    temporal_loss: torch.Tensor,
    lambda_temporal: float = 0.9,
) -> torch.Tensor:
    """
    Joint training loss: cross-entropy + lambda * temporal consistency.

    Returns:
        total_loss: Scalar training loss.
    """
    target_idx = targets.argmax(dim=1)                         # [B]
    logits_masked = logits.masked_fill(~mask.bool(), -1e9)
    ce_loss = F.cross_entropy(logits_masked, target_idx)
    return ce_loss + lambda_temporal * temporal_loss