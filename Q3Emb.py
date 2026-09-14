import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from transformers.utils import is_flash_attn_2_available


DEFAULT_MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


class Qwen3Embedding:
    """
    Lightweight wrapper around Qwen3-Embedding for encoding text into
    fixed-dimension dense vectors.
    """

    def __init__(
        self,
        model_name_or_path: str,
        model_revision: str | None = DEFAULT_MODEL_REVISION,
        use_cuda: bool = True,
        max_length: int = 8192,
    ) -> None:
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        if model_revision:
            load_kwargs["revision"] = model_revision

        if is_flash_attn_2_available() and use_cuda:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                **load_kwargs,
            )

        if use_cuda:
            self.model = self.model.cuda()
        self.model.eval()

        tokenizer_kwargs = {
            "trust_remote_code": True,
            "padding_side": "left",
        }
        if model_revision:
            tokenizer_kwargs["revision"] = model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, **tokenizer_kwargs
        )
        self.max_length = max_length

    def _last_token_pool(
        self,
        last_hidden_states: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        left_padding = (
            attention_mask[:, -1].sum() == attention_mask.shape[0]
        )
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def encode(
        self,
        sentences: list[str] | str,
        dim: int = -1,
    ) -> Tensor:
        """
        Encode one or more sentences into dense embeddings.

        Returns:
            Embedding tensor of shape [N, dim] or [N, hidden_size].
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            model_outputs = self.model(**inputs)
            output = self._last_token_pool(
                model_outputs.last_hidden_state,
                inputs["attention_mask"],
            )
            if dim > 0:
                output = output[:, :dim]

        return output
