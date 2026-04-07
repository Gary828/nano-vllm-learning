"""Tests for cache-aware scheduling in BlockManager and Scheduler."""

import pytest
from collections import deque

from nanovllm.engine.block_manager import BlockManager, Block
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams
from nanovllm.config import Config


# --- BlockManager tests ---

class TestCountCachedBlocks:

    def _make_seq(self, token_ids):
        return Sequence(token_ids, SamplingParams(temperature=0.6, max_tokens=1))

    def test_no_cache_returns_zero(self):
        bm = BlockManager(num_blocks=20, block_size=256)
        seq = self._make_seq([1] * 512)
        assert bm.count_cached_blocks(seq) == 0

    def test_after_allocate_sibling_sees_cache(self):
        """After allocating seq A, seq B sharing the same prefix sees cached blocks."""
        bm = BlockManager(num_blocks=20, block_size=256)
        prefix = list(range(256))
        suffix_a = list(range(256, 384))
        suffix_b = list(range(384, 512))

        seq_a = self._make_seq(prefix + suffix_a)
        bm.allocate(seq_a)

        seq_b = self._make_seq(prefix + suffix_b)
        assert bm.count_cached_blocks(seq_b) == 1

    def test_after_deallocate_still_cached(self):
        """After deallocating, freed blocks still have valid hash/tokens — count_cached_blocks sees them."""
        bm = BlockManager(num_blocks=20, block_size=256)
        prefix = list(range(256))

        seq_a = self._make_seq(prefix + list(range(256, 384)))
        bm.allocate(seq_a)
        bm.deallocate(seq_a)

        seq_b = self._make_seq(prefix + list(range(384, 512)))
        # Hash mapping persists, block token_ids not cleared — still a cache hit
        assert bm.count_cached_blocks(seq_b) == 1

    def test_reusable_blocks_only_counts_used(self):
        """count_reusable_blocks only counts blocks in used_block_ids."""
        bm = BlockManager(num_blocks=20, block_size=256)
        prefix = list(range(256))

        seq_a = self._make_seq(prefix + list(range(256, 384)))
        bm.allocate(seq_a)
        bm.deallocate(seq_a)

        seq_b = self._make_seq(prefix + list(range(384, 512)))
        # Block freed — not reusable (not in used_block_ids)
        assert bm.count_reusable_blocks(seq_b) == 0

        # Re-allocate to make it used again
        seq_c = self._make_seq(prefix + list(range(512, 640)))
        bm.allocate(seq_c)

        seq_d = self._make_seq(prefix + list(range(640, 768)))
        assert bm.count_reusable_blocks(seq_d) == 1


class TestCanAllocateCacheAware:

    def _make_seq(self, token_ids):
        return Sequence(token_ids, SamplingParams(temperature=0.6, max_tokens=1))

    def test_basic_allocation(self):
        bm = BlockManager(num_blocks=5, block_size=256)
        seq = self._make_seq([1] * 300)  # needs 2 blocks
        assert bm.can_allocate(seq) is True

    def test_tight_memory_without_cache_hit(self):
        """With only 1 free block and seq needing 2, can_allocate should fail."""
        bm = BlockManager(num_blocks=5, block_size=256)
        # Allocate 4 blocks to fill up
        for i in range(4):
            seq = self._make_seq([i * 1000 + j for j in range(256)])
            bm.allocate(seq)
        # 1 free block left
        new_seq = self._make_seq([9999] * 300)  # needs 2 blocks, 0 reusable
        assert bm.can_allocate(new_seq) is False

    def test_tight_memory_with_cache_hit(self):
        """With 1 free block, a seq that can reuse 1 shared block should fit (needs only 1 new)."""
        bm = BlockManager(num_blocks=5, block_size=256)
        prefix = list(range(256))
        # seq_a: 2 blocks (256 prefix + 128 suffix). Filler: 2 blocks. Total used=4, free=1.
        seq_a = self._make_seq(prefix + [9000 + j for j in range(128)])
        bm.allocate(seq_a)  # uses 2 blocks
        for i in range(1, 3):
            seq = self._make_seq([i * 1000 + j for j in range(256)])
            bm.allocate(seq)  # uses 1 block each
        # 1 free block left. New seq shares prefix with seq_a (1 reusable block).
        new_seq = self._make_seq(prefix + [8000 + j for j in range(128)])
        # needs 2 blocks, 1 reusable → needs 1 free block → should succeed
        assert bm.can_allocate(new_seq) is True


# --- Scheduler prefix sorting tests ---

class TestSortWaitingByPrefix:

    def _make_scheduler(self, num_blocks=100):
        """Create a minimal scheduler for testing."""

        class FakeConfig:
            max_num_seqs = 32
            max_num_batched_tokens = 16384
            eos = -1
            num_kvcache_blocks = num_blocks
            kvcache_block_size = 256
            cache_aware = True

        return Scheduler(FakeConfig())

    def _make_seq(self, token_ids):
        return Sequence(token_ids, SamplingParams(temperature=0.6, max_tokens=1))

    def test_groups_same_prefix_together(self):
        scheduler = self._make_scheduler()

        prefix_a = list(range(256))
        prefix_b = list(range(256, 512))

        # Add in interleaved order: A, B, A, B
        seqs = [
            self._make_seq(prefix_a + [1000]),
            self._make_seq(prefix_b + [2000]),
            self._make_seq(prefix_a + [3000]),
            self._make_seq(prefix_b + [4000]),
        ]
        for s in seqs:
            scheduler.add(s)

        scheduler._sort_waiting_by_prefix()

        # After sorting, same-prefix seqs should be adjacent
        sorted_seqs = list(scheduler.waiting)
        hash_a = BlockManager.compute_hash(prefix_a)
        hash_b = BlockManager.compute_hash(prefix_b)

        hashes = []
        for s in sorted_seqs:
            tokens = s.block(0)
            if len(tokens) == 256:
                hashes.append(BlockManager.compute_hash(tokens))
            else:
                hashes.append(-1)

        # All A's should be together and all B's together
        a_indices = [i for i, h in enumerate(hashes) if h == hash_a]
        b_indices = [i for i, h in enumerate(hashes) if h == hash_b]
        assert a_indices == list(range(a_indices[0], a_indices[0] + len(a_indices)))
        assert b_indices == list(range(b_indices[0], b_indices[0] + len(b_indices)))

    def test_single_seq_no_crash(self):
        scheduler = self._make_scheduler()
        scheduler.add(self._make_seq([1] * 300))
        scheduler._sort_waiting_by_prefix()
        assert len(scheduler.waiting) == 1

    def test_empty_no_crash(self):
        scheduler = self._make_scheduler()
        scheduler._sort_waiting_by_prefix()
        assert len(scheduler.waiting) == 0
