import torch


def estimate_quantized_block_bytes(
    num_hidden_layers: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    kv_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim
    scale_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * torch.tensor([], dtype=torch.float32).element_size()
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


def materialize_paged_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = block_tables >= 0
    if not torch.any(valid_mask):
        empty = torch.empty(0, *k_cache.shape[1:], dtype=out_dtype, device=k_cache.device)
        return empty, empty.clone(), block_tables

    used_blocks = block_tables[valid_mask].to(torch.int64)
    unique_blocks, inverse = torch.unique(used_blocks, sorted=True, return_inverse=True)

    k_int8 = torch.index_select(k_cache, 0, unique_blocks)
    v_int8 = torch.index_select(v_cache, 0, unique_blocks)
    k_scale_used = torch.index_select(k_scale, 0, unique_blocks)
    v_scale_used = torch.index_select(v_scale, 0, unique_blocks)

    k_fp = (k_int8.to(torch.float32) * k_scale_used.unsqueeze(-1)).to(out_dtype).contiguous()
    v_fp = (v_int8.to(torch.float32) * v_scale_used.unsqueeze(-1)).to(out_dtype).contiguous()

    local_block_tables = torch.full_like(block_tables, -1)
    local_block_tables[valid_mask] = inverse.to(block_tables.dtype)
    return k_fp, v_fp, local_block_tables
