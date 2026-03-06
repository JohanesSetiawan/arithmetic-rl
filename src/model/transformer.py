import torch
import torch.nn as nn

from .block import DecoderBlock
from .config import ModelConfig
from .rope import RoPE


class ArithmeticTransformer(nn.Module):
    """Decoder-only transformer for arithmetic RL training.

    Design choices:
    - No positional embedding in the token embedding layer — RoPE handles positions.
    - Weight tying between embedding and LM head: reduces parameters, improves generalisation.
    - All linear layers have bias=False (standard modern practice).
    - GPT-style init (std=0.02) for stable training from random initialisation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.embed    = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.rope     = RoPE(config.head_dim, base=config.rope_base)
        self.blocks   = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.norm_out = nn.RMSNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: lm_head and embedding share the same matrix
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:   input_ids  [B, T]
        Returns: logits    [B, T, vocab_size]
        """
        x = self.embed(input_ids)
        cos, sin = self.rope(input_ids.size(1), input_ids.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm_out(x))

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,  # [1, T]
        end_id: int,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Greedy autoregressive generation until END token or length cap."""
        self.eval()
        generated = []
        ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            next_id = self(ids)[0, -1].argmax(-1).item()
            generated.append(next_id)
            if next_id == end_id:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)

        return torch.tensor(generated, dtype=torch.long)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
