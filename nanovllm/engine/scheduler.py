from collections import deque

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.max_model_len = getattr(config, "max_model_len", 1 << 30)
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos if isinstance(config.eos, set) else {config.eos}
        self.block_size = config.kvcache_block_size
        self.running_first = getattr(config, "running_first", True)
        self._prefill_turn = False
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.cache_aware = getattr(config, "cache_aware", True)

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if len(seq) > self.max_model_len - 1:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_model_len-1 ({self.max_model_len - 1})"
            )
        self.waiting.append(seq)

    def _sort_waiting_by_prefix(self):
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

    def _take_schedulable_waiting(self, num_batched_tokens: int) -> Sequence | None:
        for i, seq in enumerate(self.waiting):
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                return None

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    continue
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if i > 0 and remaining < num_tokens:
                continue

            if i == 0:
                return self.waiting.popleft()
            self.waiting.rotate(-i)
            picked = self.waiting.popleft()
            self.waiting.rotate(i)
            return picked
        return None

    def _should_run_prefill_first(self) -> bool:
        if not self.waiting:
            return False
        if not self.running:
            return True
        if not self.running_first:
            return True
        prefill_first = self._prefill_turn
        self._prefill_turn = not self._prefill_turn
        return prefill_first

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.cache_aware and self.waiting:
            self._sort_waiting_by_prefix()

        prefill_first = self._should_run_prefill_first()

        if not prefill_first:
            scheduled_decode = []
            while self.running and len(scheduled_decode) < self.max_num_seqs:
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    seq.num_scheduled_tokens = 1
                    seq.is_prefill = False
                    self.block_manager.may_append(seq)
                    scheduled_decode.append(seq)
            if scheduled_decode:
                self.running.extendleft(reversed(scheduled_decode))
                return scheduled_decode, False

        scheduled_seqs = []
        num_batched_tokens = 0
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self._take_schedulable_waiting(num_batched_tokens)
            if seq is None:
                break

            remaining = self.max_num_batched_tokens - num_batched_tokens
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    continue
                self.block_manager.allocate(seq, num_cached_blocks)
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            if remaining < num_tokens and scheduled_seqs:
                self.waiting.appendleft(seq)
                break

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            # Publish hashes for fully-covered blocks in this prefill slice
            # so later requests in the same scheduling step can reuse them.
            self.block_manager.hash_blocks(seq)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
            else:
                self.waiting.appendleft(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        if not scheduled_seqs:
            raise RuntimeError(
                "No schedulable sequence: waiting requests exceed current token/cache budget. "
                "Try increasing max_num_batched_tokens or KV cache capacity."
            )

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id in self.eos)
                or seq.num_completion_tokens == seq.max_tokens
                or len(seq) >= self.max_model_len
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def get_cache_stats(self) -> dict:
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
