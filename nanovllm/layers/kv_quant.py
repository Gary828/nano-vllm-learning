import torch


KV_CACHE_QUANT_DTYPES = {
    "int8": torch.int8,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}


def normalize_kv_cache_quant_dtype(mode) -> str | None:
    if mode in (None, False, "", "none"):
        return None
    if mode is True:
        return "int8"
    mode = str(mode).lower()
    aliases = {"fp8": "fp8_e4m3fn"}
    mode = aliases.get(mode, mode)
    assert mode in KV_CACHE_QUANT_DTYPES, f"unsupported kv_cache_quant mode: {mode}"
    return mode


def get_kv_cache_storage_dtype(mode) -> torch.dtype | None:
    mode = normalize_kv_cache_quant_dtype(mode)
    if mode is None:
        return None
    return KV_CACHE_QUANT_DTYPES[mode]


def kv_cache_quant_uses_scale(mode) -> bool:
    return normalize_kv_cache_quant_dtype(mode) == "int8"


def estimate_quantized_block_bytes(
    num_hidden_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    quant_dtype: str = "int8",
) -> int:
    storage_dtype = get_kv_cache_storage_dtype(quant_dtype)
    assert storage_dtype is not None
    storage_bytes = torch.tensor([], dtype=storage_dtype).element_size()
    kv_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * storage_bytes
    scale_bytes = 0
    if kv_cache_quant_uses_scale(quant_dtype):
        scale_bytes = (
            2
            * num_hidden_layers
            * block_size
            * num_kv_heads
            * torch.tensor([], dtype=torch.float32).element_size()
        )
    return kv_bytes + scale_bytes


def _quantize_per_token_head(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.to(torch.float32)
    scale = x_fp32.abs().amax(dim=-1).clamp_min(1e-8) / 127.0
    q = torch.round(x_fp32 / scale.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return q, scale


def store_kvcache_int8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    valid = slot_mapping != -1
    if not torch.any(valid):
        return
    slots = slot_mapping[valid].to(torch.int64)
    key_q, key_scale = _quantize_per_token_head(key[valid])
    value_q, value_scale = _quantize_per_token_head(value[valid])
    flat_k_cache = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v_cache = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    flat_k_scale = k_scale.view(-1, k_scale.size(-1))
    flat_v_scale = v_scale.view(-1, v_scale.size(-1))
    flat_k_cache[slots] = key_q
    flat_v_cache[slots] = value_q
    flat_k_scale[slots] = key_scale
    flat_v_scale[slots] = value_scale


def store_kvcache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_dtype: str,
):
    valid = slot_mapping != -1
    if not torch.any(valid):
        return
    slots = slot_mapping[valid].to(torch.int64)
    cache_dtype = get_kv_cache_storage_dtype(quant_dtype)
    flat_k_cache = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v_cache = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    key_fp8 = key[valid].to(cache_dtype)
    value_fp8 = value[valid].to(cache_dtype)
    for idx, slot in enumerate(slots.tolist()):
        flat_k_cache[slot].copy_(key_fp8[idx])
        flat_v_cache[slot].copy_(value_fp8[idx])


def store_kvcache_quantized(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    quant_dtype: str,
):
    quant_dtype = normalize_kv_cache_quant_dtype(quant_dtype)
    if quant_dtype == "int8":
        assert k_scale is not None and v_scale is not None
        store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping)
    else:
        store_kvcache_fp8(key, value, k_cache, v_cache, slot_mapping, quant_dtype)


def materialize_paged_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    block_tables: torch.Tensor,
    out_dtype: torch.dtype,
    quant_dtype: str = "int8",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quant_dtype = normalize_kv_cache_quant_dtype(quant_dtype)
    valid_mask = block_tables >= 0
    if not torch.any(valid_mask):
        empty = torch.empty(0, *k_cache.shape[1:], dtype=out_dtype, device=k_cache.device)
        return empty, empty.clone(), block_tables

    used_blocks = block_tables[valid_mask].to(torch.int64)
    unique_blocks, inverse = torch.unique(used_blocks, sorted=True, return_inverse=True)

    if quant_dtype == "int8":
        k_quant = torch.index_select(k_cache, 0, unique_blocks)
        v_quant = torch.index_select(v_cache, 0, unique_blocks)
        assert k_scale is not None and v_scale is not None
        k_scale_used = torch.index_select(k_scale, 0, unique_blocks)
        v_scale_used = torch.index_select(v_scale, 0, unique_blocks)
        k_fp = (k_quant.to(torch.float32) * k_scale_used.unsqueeze(-1)).to(out_dtype).contiguous()
        v_fp = (v_quant.to(torch.float32) * v_scale_used.unsqueeze(-1)).to(out_dtype).contiguous()
    else:
        # CUDA kernels for float8 do not implement index_select, so cast first.
        k_fp = torch.index_select(k_cache.to(out_dtype), 0, unique_blocks).contiguous()
        v_fp = torch.index_select(v_cache.to(out_dtype), 0, unique_blocks).contiguous()

    local_block_tables = torch.full_like(block_tables, -1)
    local_block_tables[valid_mask] = inverse.to(block_tables.dtype)
    return k_fp, v_fp, local_block_tables
