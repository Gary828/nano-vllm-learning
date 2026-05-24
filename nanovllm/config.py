import os
from dataclasses import dataclass
from transformers import AutoConfig

from nanovllm.layers.kv_quant import normalize_kv_cache_quant_dtype


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kv_cache_quant: bool | str | None = False
    kv_cache_fp8_use_scale: bool = False
    kv_cache_fp8_scale_granularity: str = "per_token_head"
    kv_cache_fp8_scale_dtype: str = "fp32"
    hf_config: AutoConfig | None = None
    eos: int | set = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    cache_aware: bool = True
    running_first: bool = True

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.kv_cache_quant = normalize_kv_cache_quant_dtype(self.kv_cache_quant)
        assert self.kv_cache_fp8_scale_granularity in {"per_token_head"}
        assert self.kv_cache_fp8_scale_dtype in {"fp32"}
        if self.kv_cache_fp8_use_scale:
            assert self.kv_cache_quant in {"fp8_e4m3fn", "fp8_e5m2"}, (
                "kv_cache_fp8_use_scale requires kv_cache_quant to be fp8_e4m3fn/fp8_e5m2"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
