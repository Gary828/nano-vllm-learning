from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.cache_aware = getattr(config, 'cache_aware', True)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _sort_waiting_by_prefix(self):
        """Sort waiting queue to group sequences sharing the same prefix for better cache reuse."""
        if len(self.waiting) <= 1:
            return
        bm = self.block_manager
        def prefix_key(seq):
            if seq.num_blocks >= 1:
                tokens = seq.block(0)
                if len(tokens) == bm.block_size:
                    return bm.compute_hash(tokens)
            return -1
        self.waiting = deque(sorted(self.waiting, key=prefix_key))

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.cache_aware and self.waiting:
            self._sort_waiting_by_prefix()
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if self.cache_aware:
                predicted_new = len(seq) - self.block_manager.count_cached_blocks(seq) * self.block_manager.block_size
                predicted_new = max(predicted_new, 1)
            else:
                predicted_new = len(seq)
            if num_batched_tokens + predicted_new > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def get_cache_stats(self) -> dict:
        """Get KV cache statistics."""
        total_cached = 0
        total_tokens = 0
        for seq in self.waiting:
            total_cached += seq.num_cached_tokens
            total_tokens += len(seq)
        for seq in self.running:
            total_cached += seq.num_cached_tokens
            total_tokens += len(seq)
        cache_hit_rate = total_cached / total_tokens if total_tokens > 0 else 0.0
        return {
            "total_cached_tokens": total_cached,
            "total_tokens": total_tokens,
            "cache_hit_rate": cache_hit_rate,
        }
