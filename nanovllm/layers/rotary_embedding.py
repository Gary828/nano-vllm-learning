import math

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = dict(rope_scaling) if rope_scaling is not None else None
        self.rope_type = "default"
        if self.rope_scaling is not None:
            self.rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
        self.original_max_position_embeddings = (
            self.rope_scaling.get("original_max_position_embeddings", max_position_embeddings)
            if self.rope_scaling is not None
            else max_position_embeddings
        )

        if self.rope_type == "longrope":
            short_inv_freq, short_scale = self._compute_longrope_inv_freq(False)
            long_inv_freq, long_scale = self._compute_longrope_inv_freq(True)
            self.register_buffer(
                "cos_sin_cache_short",
                self._build_cache(max_position_embeddings, short_inv_freq, short_scale),
                persistent=False,
            )
            self.register_buffer(
                "cos_sin_cache_long",
                self._build_cache(max_position_embeddings, long_inv_freq, long_scale),
                persistent=False,
            )
        elif self.rope_type == "dynamic":
            inv_freq = self._compute_dynamic_inv_freq(max_position_embeddings)
            self.register_buffer("cos_sin_cache", self._build_cache(max_position_embeddings, inv_freq), persistent=False)
            self.max_seq_len_cached = max_position_embeddings
        else:
            inv_freq, attention_factor = self._compute_inv_freq(self.rope_type)
            self.register_buffer(
                "cos_sin_cache",
                self._build_cache(max_position_embeddings, inv_freq, attention_factor),
                persistent=False,
            )

    def _default_inv_freq(self) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

    def _compute_dynamic_inv_freq(self, seq_len: int) -> torch.Tensor:
        factor = self.rope_scaling["factor"]
        seq_len = max(seq_len, self.max_position_embeddings)
        base = self.base * ((factor * seq_len / self.max_position_embeddings) - (factor - 1)) ** (
            self.rotary_dim / (self.rotary_dim - 2)
        )
        return 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))

    def _compute_longrope_inv_freq(self, use_long_factor: bool) -> tuple[torch.Tensor, float]:
        ext_key = "long_factor" if use_long_factor else "short_factor"
        ext_factors = torch.tensor(self.rope_scaling[ext_key], dtype=torch.float)
        factor = self.rope_scaling.get("factor")
        if factor is None:
            factor = self.max_position_embeddings / self.original_max_position_embeddings
        attention_factor = self.rope_scaling.get("attention_factor")
        if attention_factor is None:
            if factor <= 1.0:
                attention_factor = 1.0
            else:
                attention_factor = math.sqrt(1 + math.log(factor) / math.log(self.original_max_position_embeddings))
        inv_freq = 1.0 / (
            ext_factors
            * (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        )
        return inv_freq, attention_factor

    def _compute_inv_freq(self, rope_type: str) -> tuple[torch.Tensor, float]:
        inv_freq = self._default_inv_freq()
        attention_factor = 1.0
        if rope_type == "default":
            return inv_freq, attention_factor
        if rope_type == "linear":
            inv_freq = inv_freq / self.rope_scaling["factor"]
            return inv_freq, attention_factor
        if rope_type == "llama3":
            factor = self.rope_scaling["factor"]
            low_freq_factor = self.rope_scaling["low_freq_factor"]
            high_freq_factor = self.rope_scaling["high_freq_factor"]
            old_context_len = self.rope_scaling["original_max_position_embeddings"]
            low_freq_wavelen = old_context_len / low_freq_factor
            high_freq_wavelen = old_context_len / high_freq_factor
            wavelen = 2 * math.pi / inv_freq
            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
            inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            return inv_freq, attention_factor
        raise NotImplementedError(f"unsupported rope_type: {rope_type}")

    def _build_cache(self, max_position_embeddings: int, inv_freq: torch.Tensor, attention_factor: float = 1.0):
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * attention_factor
        sin = freqs.sin() * attention_factor
        return torch.cat((cos, sin), dim=-1).unsqueeze_(1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Avoid host sync during CUDA graph capture. For static/default RoPE
        # paths we can index directly without reading tensor values on CPU.
        if torch.cuda.is_current_stream_capturing():
            if self.rope_type == "longrope":
                # Warmup/capture uses fixed shapes within configured max length.
                cos_sin = self.cos_sin_cache_short[positions]
            else:
                cos_sin = self.cos_sin_cache[positions]
        else:
            seq_len = positions.max().item() + 1
            if self.rope_type == "longrope":
                if seq_len > self.original_max_position_embeddings:
                    cos_sin = self.cos_sin_cache_long[positions]
                else:
                    cos_sin = self.cos_sin_cache_short[positions]
            else:
                if self.rope_type == "dynamic" and seq_len > self.max_seq_len_cached:
                    inv_freq = self._compute_dynamic_inv_freq(seq_len)
                    self.cos_sin_cache = self._build_cache(seq_len, inv_freq)
                    self.max_seq_len_cached = seq_len
                cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def _freeze_rope_scaling(rope_scaling: dict | None):
    if rope_scaling is None:
        return None
    frozen = []
    for key, value in sorted(rope_scaling.items()):
        if isinstance(value, dict):
            value = _freeze_rope_scaling(value)
        elif isinstance(value, list):
            value = tuple(value)
        frozen.append((key, value))
    return tuple(frozen)


_ROPE_CACHE: dict[tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    device = str(torch.empty(0).device)
    cache_key = (device, head_size, rotary_dim, max_position, base, _freeze_rope_scaling(rope_scaling))
    rotary_emb = _ROPE_CACHE.get(cache_key)
    if rotary_emb is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
        _ROPE_CACHE[cache_key] = rotary_emb
    return rotary_emb
