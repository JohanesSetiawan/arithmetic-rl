from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int   = 20     # character-level vocab (digits + operators + specials)
    d_model:    int   = 128    # hidden / embedding dim
    n_heads:    int   = 4      # attention heads  →  head_dim = 128 / 4 = 32
    n_layers:   int   = 4      # transformer blocks
    ffn_dim:    int   = 512    # SwiGLU inner dim (4 × d_model)
    dropout:    float = 0.1
    rope_base:  int   = 10000  # RoPE frequency base θ

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
