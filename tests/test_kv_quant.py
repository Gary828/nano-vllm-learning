import torch

from nanovllm.layers.kv_quant import (
    estimate_quantized_block_bytes,
    materialize_paged_kvcache,
    store_kvcache_int8,
)


def test_estimate_quantized_block_bytes_smaller_than_fp16():
    fp16_bytes = 2 * 4 * 256 * 8 * 64 * 2
    quant_bytes = estimate_quantized_block_bytes(4, 256, 8, 64)
    assert quant_bytes < fp16_bytes


def test_store_kvcache_int8_roundtrip_small():
    num_blocks, block_size, num_heads, head_dim = 2, 4, 2, 4
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=torch.int8)
    v_cache = torch.zeros_like(k_cache)
    k_scale = torch.zeros(num_blocks, block_size, num_heads, dtype=torch.float32)
    v_scale = torch.zeros_like(k_scale)

    key = torch.tensor([
        [[1.0, -2.0, 3.5, -4.0], [0.5, -0.5, 0.25, -0.25]],
        [[2.0, 1.0, -1.0, -2.0], [4.0, -4.0, 2.0, -2.0]],
    ])
    value = key * 0.5
    slot_mapping = torch.tensor([0, 5], dtype=torch.int32)

    store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping)

    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    k_fp, v_fp, local_block_tables = materialize_paged_kvcache(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        torch.float32,
    )

    assert torch.equal(local_block_tables, torch.tensor([[0, 1]], dtype=torch.int32))
    assert torch.allclose(k_fp[0, 0], key[0], atol=0.05)
    assert torch.allclose(v_fp[0, 0], value[0], atol=0.05)
    assert torch.allclose(k_fp[1, 1], key[1], atol=0.05)
    assert torch.allclose(v_fp[1, 1], value[1], atol=0.05)


def test_materialize_paged_kvcache_ignores_padding_and_remaps():
    k_cache = torch.arange(3 * 2 * 1 * 2, dtype=torch.int8).view(3, 2, 1, 2)
    v_cache = (k_cache + 1).clone()
    k_scale = torch.ones(3, 2, 1, dtype=torch.float32)
    v_scale = torch.ones(3, 2, 1, dtype=torch.float32)
    block_tables = torch.tensor([[2, -1, 0], [0, 2, -1]], dtype=torch.int32)

    k_fp, v_fp, local_block_tables = materialize_paged_kvcache(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        torch.float32,
    )

    assert k_fp.shape[0] == 2
    assert v_fp.shape[0] == 2
    assert torch.equal(local_block_tables, torch.tensor([[1, -1, 0], [0, 1, -1]], dtype=torch.int32))
    assert torch.allclose(k_fp[0], k_cache[0].to(torch.float32))
    assert torch.allclose(k_fp[1], k_cache[2].to(torch.float32))


def test_materialize_paged_kvcache_respects_output_dtype():
    k_cache = torch.ones(1, 2, 1, 2, dtype=torch.int8)
    v_cache = torch.ones_like(k_cache)
    k_scale = torch.full((1, 2, 1), 0.5, dtype=torch.float32)
    v_scale = torch.full((1, 2, 1), 0.25, dtype=torch.float32)
    block_tables = torch.tensor([[0]], dtype=torch.int32)

    k_fp, v_fp, _ = materialize_paged_kvcache(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_tables,
        torch.float16,
    )

    assert k_fp.dtype == torch.float16
    assert v_fp.dtype == torch.float16
    assert torch.allclose(k_fp.float(), torch.full_like(k_fp.float(), 0.5))
    assert torch.allclose(v_fp.float(), torch.full_like(v_fp.float(), 0.25))
