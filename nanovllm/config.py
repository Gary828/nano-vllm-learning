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
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
