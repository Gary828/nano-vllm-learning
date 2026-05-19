import torch
import triton
import triton.language as tl


KV_CACHE_QUANT_DTYPES = {
    "int8": torch.int8,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
FP8_CACHE_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


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


@triton.jit
def gather_dequant_int8_kernel(
    cache_ptr,
    scale_ptr,
    block_ids_ptr,
    out_ptr,
    num_rows,
    num_heads,
    head_dim,
    block_size,
    cache_stride_row,
    cache_stride_head,
    cache_stride_dim,
    scale_stride_row,
    scale_stride_head,
    out_stride_row,
    out_stride_head,
    out_stride_dim,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    chunk = tl.program_id(axis=2)
    offs_d = chunk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    block_pos = row // block_size
    token_pos = row - block_pos * block_size
    src_block = tl.load(block_ids_ptr + block_pos)
    src_row = src_block * block_size + token_pos

    cache_ptrs = cache_ptr + src_row * cache_stride_row + head * cache_stride_head + offs_d * cache_stride_dim
    q = tl.load(cache_ptrs, mask=d_mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + src_row * scale_stride_row + head * scale_stride_head)
    out = q * scale
    out_ptrs = out_ptr + row * out_stride_row + head * out_stride_head + offs_d * out_stride_dim
    tl.store(out_ptrs, out, mask=d_mask)


@triton.jit
def gather_cast_fp8_kernel(
    cache_ptr,
    block_ids_ptr,
    out_ptr,
    head_dim,
    block_size,
    cache_stride_row,
    cache_stride_head,
    cache_stride_dim,
    out_stride_row,
    out_stride_head,
    out_stride_dim,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    chunk = tl.program_id(axis=2)
    offs_d = chunk * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    block_pos = row // block_size
    token_pos = row - block_pos * block_size
    src_block = tl.load(block_ids_ptr + block_pos)
    src_row = src_block * block_size + token_pos

    cache_ptrs = cache_ptr + src_row * cache_stride_row + head * cache_stride_head + offs_d * cache_stride_dim
    x = tl.load(cache_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    out_ptrs = out_ptr + row * out_stride_row + head * out_stride_head + offs_d * out_stride_dim
    tl.store(out_ptrs, x, mask=d_mask)


@triton.jit
def quantize_store_int8_kernel(
    in_ptr,
    in_stride_row,
    in_stride_head,
    in_stride_dim,
    slots_ptr,
    cache_ptr,
    cache_stride_row,
    cache_stride_head,
    cache_stride_dim,
    scale_ptr,
    scale_stride_row,
    scale_stride_head,
    head_dim,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    slot = tl.load(slots_ptr + row)
    if slot < 0:
        return

    x_ptrs = (
        in_ptr
        + row * in_stride_row
        + head * in_stride_head
        + offs_d * in_stride_dim
    )
    x = tl.load(x_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    abs_x = tl.abs(x)
    max_abs = tl.max(abs_x, axis=0)
    scale = tl.maximum(max_abs / 127.0, 1e-8)
    inv_scale = 1.0 / scale
    q = x * inv_scale
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)
    q = tl.extra.cuda.libdevice.rint(q).to(tl.int8)

    cache_ptrs = (
        cache_ptr
        + slot * cache_stride_row
        + head * cache_stride_head
        + offs_d * cache_stride_dim
    )
    tl.store(cache_ptrs, q, mask=d_mask)
    scale_loc = scale_ptr + slot * scale_stride_row + head * scale_stride_head
    tl.store(scale_loc, scale)


@triton.jit
def quantize_store_int8_pair_kernel(
    k_in_ptr,
    k_in_stride_row,
    k_in_stride_head,
    k_in_stride_dim,
    v_in_ptr,
    v_in_stride_row,
    v_in_stride_head,
    v_in_stride_dim,
    slots_ptr,
    k_cache_ptr,
    k_cache_stride_row,
    k_cache_stride_head,
    k_cache_stride_dim,
    v_cache_ptr,
    v_cache_stride_row,
    v_cache_stride_head,
    v_cache_stride_dim,
    k_scale_ptr,
    k_scale_stride_row,
    k_scale_stride_head,
    v_scale_ptr,
    v_scale_stride_row,
    v_scale_stride_head,
    head_dim,
    block_size_tokens,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    slot = tl.load(slots_ptr + row)
    if slot < 0:
        return

    k_ptrs = (
        k_in_ptr
        + row * k_in_stride_row
        + head * k_in_stride_head
        + offs_d * k_in_stride_dim
    )
    v_ptrs = (
        v_in_ptr
        + row * v_in_stride_row
        + head * v_in_stride_head
        + offs_d * v_in_stride_dim
    )
    kx = tl.load(k_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    vx = tl.load(v_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    k_abs = tl.abs(kx)
    v_abs = tl.abs(vx)
    k_scale = tl.maximum(tl.max(k_abs, axis=0) / 127.0, 1e-8)
    v_scale = tl.maximum(tl.max(v_abs, axis=0) / 127.0, 1e-8)
    k_inv_scale = 1.0 / k_scale
    v_inv_scale = 1.0 / v_scale

    kq = kx * k_inv_scale
    vq = vx * v_inv_scale
    kq = tl.maximum(tl.minimum(kq, 127.0), -127.0)
    vq = tl.maximum(tl.minimum(vq, 127.0), -127.0)
    kq = tl.extra.cuda.libdevice.rint(kq).to(tl.int8)
    vq = tl.extra.cuda.libdevice.rint(vq).to(tl.int8)

    k_cache_ptrs = (
        k_cache_ptr
        + slot * k_cache_stride_row
        + head * k_cache_stride_head
        + offs_d * k_cache_stride_dim
    )
    v_cache_ptrs = (
        v_cache_ptr
        + slot * v_cache_stride_row
        + head * v_cache_stride_head
        + offs_d * v_cache_stride_dim
    )
    tl.store(k_cache_ptrs, kq, mask=d_mask)
    tl.store(v_cache_ptrs, vq, mask=d_mask)

    k_scale_loc = k_scale_ptr + slot * k_scale_stride_row + head * k_scale_stride_head
    v_scale_loc = v_scale_ptr + slot * v_scale_stride_row + head * v_scale_stride_head
    tl.store(k_scale_loc, k_scale)
    tl.store(v_scale_loc, v_scale)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=2),
    ],
    key=["head_dim", "block_size_tokens"],
)
@triton.jit
def store_fp8_pair_kernel(
    k_in_ptr,
    k_in_stride_row,
    k_in_stride_head,
    k_in_stride_dim,
    v_in_ptr,
    v_in_stride_row,
    v_in_stride_head,
    v_in_stride_dim,
    slots_ptr,
    k_cache_ptr,
    k_cache_stride_row,
    k_cache_stride_head,
    k_cache_stride_dim,
    v_cache_ptr,
    v_cache_stride_row,
    v_cache_stride_head,
    v_cache_stride_dim,
    head_dim,
    block_size_tokens,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(axis=0)
    head = tl.program_id(axis=1)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    slot = tl.load(slots_ptr + row)
    if slot < 0:
        return

    k_ptrs = (
        k_in_ptr
        + row * k_in_stride_row
        + head * k_in_stride_head
        + offs_d * k_in_stride_dim
    )
    v_ptrs = (
        v_in_ptr
        + row * v_in_stride_row
        + head * v_in_stride_head
        + offs_d * v_in_stride_dim
    )
    kx = tl.load(k_ptrs, mask=d_mask, other=0.0)
    vx = tl.load(v_ptrs, mask=d_mask, other=0.0)

    k_cache_ptrs = (
        k_cache_ptr
        + slot * k_cache_stride_row
        + head * k_cache_stride_head
        + offs_d * k_cache_stride_dim
    )
    v_cache_ptrs = (
        v_cache_ptr
        + slot * v_cache_stride_row
        + head * v_cache_stride_head
        + offs_d * v_cache_stride_dim
    )
    tl.store(k_cache_ptrs, kx, mask=d_mask)
    tl.store(v_cache_ptrs, vx, mask=d_mask)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _gather_dequant_int8_fused(
    cache: torch.Tensor,
    scale: torch.Tensor,
    block_ids: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    if not cache.is_cuda or not scale.is_cuda or not block_ids.is_cuda:
        return None
    if not cache.is_contiguous() or not scale.is_contiguous() or not block_ids.is_contiguous():
        return None
    if cache.dtype != torch.int8 or scale.dtype != torch.float32:
        return None
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None

    num_selected = int(block_ids.numel())
    block_size = int(cache.size(1))
    num_heads = int(cache.size(2))
    head_dim = int(cache.size(3))
    out = torch.empty((num_selected, block_size, num_heads, head_dim), dtype=out_dtype, device=cache.device)
    if num_selected == 0:
        return out

    rows = num_selected * block_size
    block_d = min(256, _next_power_of_2(head_dim))
    grid = (rows, num_heads, triton.cdiv(head_dim, block_d))
    cache_2d = cache.view(-1, num_heads, head_dim)
    scale_2d = scale.view(-1, num_heads)
    out_3d = out.view(rows, num_heads, head_dim)
    gather_dequant_int8_kernel[grid](
        cache_2d,
        scale_2d,
        block_ids,
        out_3d,
        rows,
        num_heads,
        head_dim,
        block_size,
        cache_2d.stride(0),
        cache_2d.stride(1),
        cache_2d.stride(2),
        scale_2d.stride(0),
        scale_2d.stride(1),
        out_3d.stride(0),
        out_3d.stride(1),
        out_3d.stride(2),
        BLOCK_D=block_d,
    )
    return out


def _gather_cast_fp8_fused(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    if not cache.is_cuda or not block_ids.is_cuda:
        return None
    if not cache.is_contiguous() or not block_ids.is_contiguous():
        return None
    if cache.dtype not in FP8_CACHE_DTYPES:
        return None
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None

    num_selected = int(block_ids.numel())
    block_size = int(cache.size(1))
    num_heads = int(cache.size(2))
    head_dim = int(cache.size(3))
    out = torch.empty((num_selected, block_size, num_heads, head_dim), dtype=out_dtype, device=cache.device)
    if num_selected == 0:
        return out

    rows = num_selected * block_size
    block_d = min(256, _next_power_of_2(head_dim))
    grid = (rows, num_heads, triton.cdiv(head_dim, block_d))
    cache_3d = cache.view(-1, num_heads, head_dim)
    out_3d = out.view(rows, num_heads, head_dim)
    try:
        gather_cast_fp8_kernel[grid](
            cache_3d,
            block_ids,
            out_3d,
            head_dim,
            block_size,
            cache_3d.stride(0),
            cache_3d.stride(1),
            cache_3d.stride(2),
            out_3d.stride(0),
            out_3d.stride(1),
            out_3d.stride(2),
            BLOCK_D=block_d,
        )
    except Exception:
        return None
    return out


def _quantize_store_int8_fused(
    x: torch.Tensor,
    slots: torch.Tensor,
    cache: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    if not x.is_cuda or not slots.is_cuda or not cache.is_cuda or not scale.is_cuda:
        return False
    if (
        x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or slots.dtype != torch.int64
        or cache.dtype != torch.int8
        or scale.dtype != torch.float32
    ):
        return False
    if x.dim() != 3:
        return False
    if not x.is_contiguous():
        x = x.contiguous()
    if not slots.is_contiguous():
        slots = slots.contiguous()

    n, num_heads, head_dim = x.shape
    if n == 0:
        return True

    block_d = min(256, _next_power_of_2(head_dim))
    grid = (n, num_heads)
    cache_2d = cache.view(-1, cache.size(-2), cache.size(-1))
    scale_2d = scale.view(-1, scale.size(-1))
    quantize_store_int8_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        slots,
        cache_2d,
        cache_2d.stride(0),
        cache_2d.stride(1),
        cache_2d.stride(2),
        scale_2d,
        scale_2d.stride(0),
        scale_2d.stride(1),
        head_dim,
        BLOCK_D=block_d,
    )
    return True


def _store_fp8_pair_fused(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> bool:
    if (
        not key.is_cuda
        or not value.is_cuda
        or not slots.is_cuda
        or not k_cache.is_cuda
        or not v_cache.is_cuda
    ):
        return False
    if (
        key.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or slots.dtype not in (torch.int32, torch.int64)
        or k_cache.dtype not in FP8_CACHE_DTYPES
        or v_cache.dtype not in FP8_CACHE_DTYPES
    ):
        return False
    if key.shape != value.shape or key.dim() != 3:
        return False
    if slots.dim() != 1 or slots.numel() != key.size(0):
        return False

    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    if not slots.is_contiguous():
        slots = slots.contiguous()

    n, num_heads, head_dim = key.shape
    if n == 0:
        return True

    flat_k_cache = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v_cache = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    grid = (n, num_heads)
    try:
        store_fp8_pair_kernel[grid](
            key,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value,
            value.stride(0),
            value.stride(1),
            value.stride(2),
            slots,
            flat_k_cache,
            flat_k_cache.stride(0),
            flat_k_cache.stride(1),
            flat_k_cache.stride(2),
            flat_v_cache,
            flat_v_cache.stride(0),
            flat_v_cache.stride(1),
            flat_v_cache.stride(2),
            head_dim,
            int(k_cache.size(1)),
        )
    except Exception:
        return False
    return True


def _workspace_view(
    workspace: dict,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= max(int(dim), 1)
    buf = workspace.get(name)
    if buf is None or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(numel, dtype=dtype, device=device)
    elif buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=device)
    workspace[name] = buf
    return buf[:numel].view(*shape)


def _quantize_store_int8_pair_fused(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> bool:
    # Disabled due to correctness instability on real workloads.
    # Keep fallback path as the authoritative implementation.
    return False


def build_local_block_tables(block_tables: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_mask = block_tables >= 0
    local_block_tables = torch.full_like(block_tables, -1)
    if not torch.any(valid_mask):
        empty = torch.empty(0, dtype=torch.int64, device=block_tables.device)
        return valid_mask, empty, local_block_tables
    used_blocks = block_tables[valid_mask].to(torch.int64)
    unique_blocks, inverse = torch.unique(used_blocks, sorted=True, return_inverse=True)
    local_block_tables[valid_mask] = inverse.to(block_tables.dtype)
    return valid_mask, unique_blocks, local_block_tables


def materialize_quantized_blocks(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    block_ids: torch.Tensor,
    out_dtype: torch.dtype,
    quant_dtype: str = "int8",
) -> tuple[torch.Tensor, torch.Tensor]:
    quant_dtype = normalize_kv_cache_quant_dtype(quant_dtype)
    if block_ids.numel() == 0:
        empty = torch.empty(0, *k_cache.shape[1:], dtype=out_dtype, device=k_cache.device)
        return empty, empty.clone()
    if quant_dtype == "int8":
        assert k_scale is not None and v_scale is not None
        k_fp = _gather_dequant_int8_fused(k_cache, k_scale, block_ids, out_dtype)
        v_fp = _gather_dequant_int8_fused(v_cache, v_scale, block_ids, out_dtype)
        if k_fp is not None and v_fp is not None:
            return k_fp.contiguous(), v_fp.contiguous()
        k_fp32 = torch.index_select(k_cache, 0, block_ids).to(torch.float32)
        v_fp32 = torch.index_select(v_cache, 0, block_ids).to(torch.float32)
        k_scale_used = torch.index_select(k_scale, 0, block_ids)
        v_scale_used = torch.index_select(v_scale, 0, block_ids)
        k_fp32.mul_(k_scale_used.unsqueeze(-1))
        v_fp32.mul_(v_scale_used.unsqueeze(-1))
        k_fp = k_fp32.to(out_dtype).contiguous()
        v_fp = v_fp32.to(out_dtype).contiguous()
        return k_fp, v_fp

    k_fp = _gather_cast_fp8_fused(k_cache, block_ids, out_dtype)
    v_fp = _gather_cast_fp8_fused(v_cache, block_ids, out_dtype)
    if k_fp is not None and v_fp is not None:
        return k_fp.contiguous(), v_fp.contiguous()
    # CUDA kernels for float8 do not implement index_select, so cast first.
    k_fp = torch.index_select(k_cache.to(out_dtype), 0, block_ids).contiguous()
    v_fp = torch.index_select(v_cache.to(out_dtype), 0, block_ids).contiguous()
    return k_fp, v_fp


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
    workspace: dict | None = None,
):
    if slot_mapping.numel() == 0:
        return
    slots = slot_mapping if slot_mapping.is_contiguous() else slot_mapping.contiguous()
    key_valid = key
    value_valid = value
    if slots.dtype == torch.int64:
        if workspace is None:
            slots = slots.to(torch.int32)
        else:
            slots_i32 = _workspace_view(workspace, "slots_i32_cast", tuple(slots.shape), torch.int32, slots.device)
            slots_i32.copy_(slots)
            slots = slots_i32
    elif slots.dtype != torch.int32:
        if workspace is None:
            slots = slots.to(torch.int32)
        else:
            slots_i32 = _workspace_view(workspace, "slots_i32_cast", tuple(slots.shape), torch.int32, slots.device)
            slots_i32.copy_(slots.to(torch.int32))
            slots = slots_i32

    if _quantize_store_int8_pair_fused(
        key_valid,
        value_valid,
        slots,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
    ):
        return

    flat_k_cache = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v_cache = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    flat_k_scale = k_scale.view(-1, k_scale.size(-1))
    flat_v_scale = v_scale.view(-1, v_scale.size(-1))

    valid = slot_mapping != -1
    if not torch.any(valid):
        return
    if bool(torch.all(valid)):
        slots_l = slot_mapping.to(torch.int64)
        key_valid = key
        value_valid = value
    else:
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        if workspace is None:
            slots_l = torch.index_select(slot_mapping, 0, valid_idx).to(torch.int64)
            key_valid = torch.index_select(key, 0, valid_idx)
            value_valid = torch.index_select(value, 0, valid_idx)
        else:
            n_valid = int(valid_idx.numel())
            slots_tmp = _workspace_view(
                workspace,
                "slots_fallback",
                (n_valid,),
                slot_mapping.dtype,
                slot_mapping.device,
            )
            key_valid = _workspace_view(
                workspace,
                "key_valid",
                (n_valid, key.size(1), key.size(2)),
                key.dtype,
                key.device,
            )
            value_valid = _workspace_view(
                workspace,
                "value_valid",
                (n_valid, value.size(1), value.size(2)),
                value.dtype,
                value.device,
            )
            torch.index_select(slot_mapping, 0, valid_idx, out=slots_tmp)
            torch.index_select(key, 0, valid_idx, out=key_valid)
            torch.index_select(value, 0, valid_idx, out=value_valid)
            slots_l = slots_tmp.to(torch.int64)
    key_q, key_scale = _quantize_per_token_head(key_valid)
    value_q, value_scale = _quantize_per_token_head(value_valid)
    flat_k_cache[slots_l] = key_q
    flat_v_cache[slots_l] = value_q
    flat_k_scale[slots_l] = key_scale
    flat_v_scale[slots_l] = value_scale


def store_kvcache_fp8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_dtype: str,
    workspace: dict | None = None,
):
    if slot_mapping.numel() == 0:
        return
    slots = slot_mapping if slot_mapping.is_contiguous() else slot_mapping.contiguous()
    if slots.dtype == torch.int64:
        if workspace is None:
            slots = slots.to(torch.int32)
        else:
            slots_i32 = _workspace_view(workspace, "slots_i32_cast_fp8", tuple(slots.shape), torch.int32, slots.device)
            slots_i32.copy_(slots)
            slots = slots_i32
    elif slots.dtype != torch.int32:
        if workspace is None:
            slots = slots.to(torch.int32)
        else:
            slots_i32 = _workspace_view(
                workspace,
                "slots_i32_cast_fp8",
                tuple(slots.shape),
                torch.int32,
                slots.device,
            )
            slots_i32.copy_(slots.to(torch.int32))
            slots = slots_i32

    if _store_fp8_pair_fused(key, value, slots, k_cache, v_cache):
        return

    valid = slot_mapping != -1
    if not torch.any(valid):
        return
    cache_dtype = get_kv_cache_storage_dtype(quant_dtype)
    flat_k_cache = k_cache.view(-1, k_cache.size(-2), k_cache.size(-1))
    flat_v_cache = v_cache.view(-1, v_cache.size(-2), v_cache.size(-1))
    if bool(torch.all(valid)):
        slots_l = slot_mapping.to(torch.int64)
        key_valid = key
        value_valid = value
    else:
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        if workspace is None:
            slots_l = torch.index_select(slot_mapping, 0, valid_idx).to(torch.int64)
            key_valid = torch.index_select(key, 0, valid_idx)
            value_valid = torch.index_select(value, 0, valid_idx)
        else:
            n_valid = int(valid_idx.numel())
            slots_tmp = _workspace_view(
                workspace,
                "slots_fallback_fp8",
                (n_valid,),
                slot_mapping.dtype,
                slot_mapping.device,
            )
            key_valid = _workspace_view(
                workspace,
                "key_valid_fp8",
                (n_valid, key.size(1), key.size(2)),
                key.dtype,
                key.device,
            )
            value_valid = _workspace_view(
                workspace,
                "value_valid_fp8",
                (n_valid, value.size(1), value.size(2)),
                value.dtype,
                value.device,
            )
            torch.index_select(slot_mapping, 0, valid_idx, out=slots_tmp)
            torch.index_select(key, 0, valid_idx, out=key_valid)
            torch.index_select(value, 0, valid_idx, out=value_valid)
            slots_l = slots_tmp.to(torch.int64)
    key_fp8 = key_valid.to(cache_dtype)
    value_fp8 = value_valid.to(cache_dtype)
    for idx, slot in enumerate(slots_l.tolist()):
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
    workspace: dict | None = None,
):
    quant_dtype = normalize_kv_cache_quant_dtype(quant_dtype)
    if quant_dtype == "int8":
        assert k_scale is not None and v_scale is not None
        store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping, workspace=workspace)
    else:
        store_kvcache_fp8(key, value, k_cache, v_cache, slot_mapping, quant_dtype, workspace=workspace)


def materialize_paged_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    block_tables: torch.Tensor,
    out_dtype: torch.dtype,
    quant_dtype: str = "int8",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, unique_blocks, local_block_tables = build_local_block_tables(block_tables)
    if unique_blocks.numel() == 0:
        empty = torch.empty(0, *k_cache.shape[1:], dtype=out_dtype, device=k_cache.device)
        return empty, empty.clone(), block_tables

    k_fp, v_fp = materialize_quantized_blocks(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        unique_blocks,
        out_dtype,
        quant_dtype,
    )
    return k_fp, v_fp, local_block_tables
