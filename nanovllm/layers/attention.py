import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.kv_quant import (
    materialize_paged_kvcache,
    materialize_quantized_blocks,
    store_kvcache_quantized,
)
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n_tokens, num_heads, head_dim = key.shape
    hidden_dim = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == hidden_dim and v_cache.stride(1) == hidden_dim
    assert slot_mapping.numel() == n_tokens
    store_kvcache_kernel[(n_tokens,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        hidden_dim,
    )


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_quant = None
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = torch.tensor([])
        self._kvq_workspace = {}
        self._cached_block_tables = None
        self._cached_unique_blocks = None
        self._cached_materialized = None

    def _invalidate_materialized_cache(self):
        self._cached_block_tables = None
        self._cached_unique_blocks = None
        self._cached_materialized = None

    def _store_paged_kv_cache(self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor):
        if self.kv_cache_quant:
            store_kvcache_quantized(
                k,
                v,
                self.k_cache,
                self.v_cache,
                self.k_scale if self.k_scale.numel() else None,
                self.v_scale if self.v_scale.numel() else None,
                slot_mapping,
                self.kv_cache_quant,
                workspace=self._kvq_workspace,
            )
        else:
            store_kvcache(k, v, self.k_cache, self.v_cache, slot_mapping)
        self._invalidate_materialized_cache()

    def _materialize_cached_kv(self, block_tables: torch.Tensor, out_dtype: torch.dtype):
        context = get_context()
        if self.kv_cache_quant:
            if (
                not context.is_prefill
                and context.unique_blocks is not None
                and context.local_block_tables is not None
            ):
                unique_blocks = context.unique_blocks
                local_block_tables = context.local_block_tables
                if unique_blocks.numel() == 0:
                    empty = torch.empty(0, *self.k_cache.shape[1:], dtype=out_dtype, device=self.k_cache.device)
                    self._invalidate_materialized_cache()
                    return empty, empty.clone(), local_block_tables
                k_fp, v_fp = materialize_quantized_blocks(
                    self.k_cache,
                    self.v_cache,
                    self.k_scale if self.k_scale.numel() else None,
                    self.v_scale if self.v_scale.numel() else None,
                    unique_blocks,
                    out_dtype,
                    self.kv_cache_quant,
                )
                self._invalidate_materialized_cache()
                return k_fp, v_fp, local_block_tables
            materialized = materialize_paged_kvcache(
                self.k_cache,
                self.v_cache,
                self.k_scale if self.k_scale.numel() else None,
                self.v_scale if self.v_scale.numel() else None,
                block_tables,
                out_dtype,
                self.kv_cache_quant,
            )
            self._invalidate_materialized_cache()
            return materialized

        self._invalidate_materialized_cache()
        return self.k_cache, self.v_cache, block_tables

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            self._store_paged_kv_cache(k, v, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v, block_tables = self._materialize_cached_kv(context.block_tables, q.dtype)
            else:
                block_tables = None
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=block_tables,
            )
        else:
            k_cache, v_cache, block_tables = self._materialize_cached_kv(context.block_tables, q.dtype)
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o
