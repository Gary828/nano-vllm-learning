import torch

from nanovllm.layers.attention import Attention
from nanovllm.utils.context import set_context, reset_context


def test_decode_store_invalidates_materialized_cache():
    attn = Attention(num_heads=1, head_dim=2, scale=1.0, num_kv_heads=1)
    attn.kv_cache_quant = "int8"
    attn.k_cache = torch.zeros(1, 4, 1, 2, dtype=torch.int8)
    attn.v_cache = torch.zeros_like(attn.k_cache)
    attn.k_scale = torch.ones(1, 4, 1, dtype=torch.float32)
    attn.v_scale = torch.ones_like(attn.k_scale)

    block_tables = torch.tensor([[0]], dtype=torch.int32)
    slot_mapping = torch.tensor([0], dtype=torch.int32)
    context_lens = torch.tensor([1], dtype=torch.int32)

    try:
        # step 1: store + materialize
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        k1 = torch.tensor([[[1.0, 2.0]]], dtype=torch.float32)
        v1 = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)
        attn._store_paged_kv_cache(k1, v1, slot_mapping)
        k_fp_1, _, _ = attn._materialize_cached_kv(block_tables, torch.float32)
        old_value = k_fp_1[0, 0, 0].clone()

        # step 2: overwrite same slot with new values
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        k2 = torch.tensor([[[7.0, 8.0]]], dtype=torch.float32)
        v2 = torch.tensor([[[9.0, 10.0]]], dtype=torch.float32)
        attn._store_paged_kv_cache(k2, v2, slot_mapping)

        # store must invalidate cached materialized view
        assert attn._cached_materialized is None
        assert attn._cached_block_tables is None

        k_fp_2, _, _ = attn._materialize_cached_kv(block_tables, torch.float32)
        new_value = k_fp_2[0, 0, 0]

        # should observe updated content rather than stale materialized values
        assert not torch.allclose(new_value, old_value, atol=1e-4)
        assert torch.allclose(new_value, torch.tensor([7.0, 8.0]), atol=0.2)
    finally:
        reset_context()
