import torch

from nanovllm.layers.kv_quant import (
    build_local_block_tables,
    estimate_quantized_block_bytes,
    materialize_quantized_blocks,
    normalize_kv_cache_quant_dtype,
    materialize_paged_kvcache,
    store_kvcache_fp8,
    store_kvcache_int8,
)


def test_estimate_quantized_block_bytes_smaller_than_fp16():
    fp16_bytes = 2 * 4 * 256 * 8 * 64 * 2
    quant_bytes = estimate_quantized_block_bytes(4, 256, 8, 64, "int8")
    assert quant_bytes < fp16_bytes


def test_estimate_fp8_block_bytes_smaller_than_int8():
    int8_bytes = estimate_quantized_block_bytes(4, 256, 8, 64, "int8")
    fp8_bytes = estimate_quantized_block_bytes(4, 256, 8, 64, "fp8_e4m3fn")
    assert fp8_bytes < int8_bytes


def test_normalize_kv_cache_quant_dtype():
    assert normalize_kv_cache_quant_dtype(False) is None
    assert normalize_kv_cache_quant_dtype(None) is None
    assert normalize_kv_cache_quant_dtype(True) == "int8"
    assert normalize_kv_cache_quant_dtype("fp8") == "fp8_e4m3fn"
    assert normalize_kv_cache_quant_dtype("fp8_e5m2") == "fp8_e5m2"


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


def test_store_kvcache_fp8_roundtrip_small():
    num_blocks, block_size, num_heads, head_dim = 2, 4, 2, 4
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros_like(k_cache)

    key = torch.tensor([
        [[1.0, -2.0, 3.5, -4.0], [0.5, -0.5, 0.25, -0.25]],
        [[2.0, 1.0, -1.0, -2.0], [4.0, -4.0, 2.0, -2.0]],
    ])
    value = key * 0.5
    slot_mapping = torch.tensor([0, 5], dtype=torch.int32)

    store_kvcache_fp8(key, value, k_cache, v_cache, slot_mapping, "fp8_e4m3fn")

    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    k_fp, v_fp, local_block_tables = materialize_paged_kvcache(
        k_cache,
        v_cache,
        None,
        None,
        block_tables,
        torch.float32,
        "fp8_e4m3fn",
    )

    assert torch.equal(local_block_tables, torch.tensor([[0, 1]], dtype=torch.int32))
    assert torch.allclose(k_fp[0, 0], key[0], atol=0.25)
    assert torch.allclose(v_fp[0, 0], value[0], atol=0.25)
    assert torch.allclose(k_fp[1, 1], key[1], atol=0.25)
    assert torch.allclose(v_fp[1, 1], value[1], atol=0.25)


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
        "int8",
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
        "int8",
    )

    assert k_fp.dtype == torch.float16
    assert v_fp.dtype == torch.float16
    assert torch.allclose(k_fp.float(), torch.full_like(k_fp.float(), 0.5))
    assert torch.allclose(v_fp.float(), torch.full_like(v_fp.float(), 0.25))


def test_build_local_block_tables():
    block_tables = torch.tensor([[2, -1, 0], [0, 2, -1]], dtype=torch.int32)
    valid_mask, unique_blocks, local_block_tables = build_local_block_tables(block_tables)

    assert torch.equal(valid_mask, torch.tensor([[True, False, True], [True, True, False]]))
    assert torch.equal(unique_blocks, torch.tensor([0, 2], dtype=torch.int64))
    assert torch.equal(local_block_tables, torch.tensor([[1, -1, 0], [0, 1, -1]], dtype=torch.int32))


def test_materialize_quantized_blocks_int8_subset():
    k_cache = torch.arange(4 * 2 * 1 * 2, dtype=torch.int8).view(4, 2, 1, 2)
    v_cache = (k_cache + 1).clone()
    k_scale = torch.ones(4, 2, 1, dtype=torch.float32)
    v_scale = torch.ones(4, 2, 1, dtype=torch.float32)
    block_ids = torch.tensor([3, 1], dtype=torch.int64)

    k_fp, v_fp = materialize_quantized_blocks(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_ids,
        torch.float32,
        "int8",
    )

    assert torch.allclose(k_fp[0], k_cache[3].float())
    assert torch.allclose(k_fp[1], k_cache[1].float())
    assert torch.allclose(v_fp[0], v_cache[3].float())
    assert torch.allclose(v_fp[1], v_cache[1].float())


def test_materialize_quantized_blocks_int8_bfloat16_output():
    k_cache = torch.tensor(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ],
        dtype=torch.int8,
    )
    v_cache = (k_cache * 2).to(torch.int8)
    k_scale = torch.full((2, 2, 2), 0.5, dtype=torch.float32)
    v_scale = torch.full((2, 2, 2), 0.25, dtype=torch.float32)
    block_ids = torch.tensor([1, 0], dtype=torch.int64)

    k_fp, v_fp = materialize_quantized_blocks(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_ids,
        torch.bfloat16,
        "int8",
    )

    assert k_fp.dtype == torch.bfloat16
    assert v_fp.dtype == torch.bfloat16
    expected_k = torch.index_select(k_cache, 0, block_ids).float() * 0.5
    expected_v = torch.index_select(v_cache, 0, block_ids).float() * 0.25
    assert torch.allclose(k_fp.float(), expected_k, atol=1e-3)
    assert torch.allclose(v_fp.float(), expected_v, atol=1e-3)
