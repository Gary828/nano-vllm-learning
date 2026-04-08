#!/usr/bin/env python3
"""
nano-vllm + ContextPilot Realistic Benchmark

This benchmark tests ContextPilot's optimization on realistic scenarios:
1. RAG Retrieval: Multiple queries retrieving overlapping document chunks
2. Multi-turn Chat: Each turn appends to conversation history
3. Batch Inference: Multiple requests sharing system prompts

Unlike simple synthetic data, these scenarios reflect real-world prefix caching patterns.

Usage:
    python bench_realistic.py --scenario rag
    python bench_realistic.py --scenario multi_turn
    python bench_realistic.py --scenario batch
    python bench_realistic.py --all
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from random import randint, sample, seed
from collections import defaultdict

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(__file__))


@dataclass
class BenchmarkResult:
    scenario: str
    num_prompts: int
    total_tokens: int
    without_optimization_time: float
    with_optimization_time: float
    without_cache_hit_rate: float
    with_cache_hit_rate: float
    speedup: float
    optimization_overhead: float  # Time spent on reordering


class RealisticPromptGenerator:
    """
    Generates realistic prompts that reflect real-world prefix caching scenarios.

    Unlike random tokens, these prompts have controlled overlap patterns
    that simulate actual RAG, multi-turn chat, and batch inference workloads.
    """

    def __init__(self, seed_value: int = 42):
        self.seed_value = seed_value
        seed(seed_value)

    def generate_rag_prompts(
        self,
        num_queries: int = 32,
        chunks_per_query: int = 8,
        chunk_size: int = 128,
        num_shared_chunks: int = 4,
        vocab_size: int = 50000,
    ) -> tuple[List[List[int]], Dict[int, List[int]]]:
        """
        Generate RAG-style prompts where queries retrieve overlapping document chunks.

        Args:
            num_queries: Number of queries (e.g., 32)
            chunks_per_query: Chunks retrieved per query (e.g., 8)
            chunk_size: Tokens per chunk (e.g., 128)
            num_shared_chunks: Chunks shared across queries (e.g., 4)
            vocab_size: Vocabulary size for token IDs

        Returns:
            prompts: List of token sequences (prompts ready for inference)
            query_to_chunks: Mapping of query index to chunk indices
        """
        # Generate a shared document corpus
        # Each chunk has a "type" prefix + random content
        num_chunks = num_queries * chunks_per_query
        chunks = []

        # Shared chunks (common knowledge across queries)
        shared_chunk_ids = list(range(num_shared_chunks))
        for i in shared_chunk_ids:
            # Each shared chunk has a type prefix + content
            type_prefix = [i]  # Chunk type ID
            content = [num_shared_chunks + randint(0, vocab_size) for _ in range(chunk_size - 1)]
            chunks.append(type_prefix + content)

        # Query-specific chunks
        for i in range(num_queries):
            for j in range(chunks_per_query - num_shared_chunks):
                chunk_id = num_shared_chunks + i * (chunks_per_query - num_shared_chunks) + j
                # Embed chunk_id in the content to make it unique
                content = [chunk_id] + [randint(0, vocab_size) for _ in range(chunk_size - 1)]
                chunks.append(content)

        # Generate prompts by concatenating chunks for each query
        prompts = []
        query_to_chunks = {}

        for q in range(num_queries):
            # Each query retrieves chunks_per_query chunks
            # Include all shared chunks + query-specific chunks
            retrieved = list(shared_chunk_ids)
            for j in range(num_shared_chunks, chunks_per_query):
                chunk_idx = num_shared_chunks + q * (chunks_per_query - num_shared_chunks) + (j - num_shared_chunks)
                if chunk_idx < len(chunks):
                    retrieved.append(chunk_idx)

            query_to_chunks[q] = retrieved

            # Concatenate chunks to form prompt
            prompt = []
            for chunk_idx in retrieved:
                prompt.extend(chunks[chunk_idx])
            prompts.append(prompt)

        return prompts, query_to_chunks

    def generate_multi_turn_chat(
        self,
        num_sessions: int = 16,
        turns_per_session: int = 8,
        system_prompt_len: int = 256,
        context_len: int = 512,
        vocab_size: int = 50000,
    ) -> tuple[List[List[int]], Dict[int, List[int]]]:
        """
        Generate multi-turn chat prompts where each turn shares conversation history.

        Args:
            num_sessions: Number of parallel chat sessions
            turns_per_session: Turns per session (each turn adds to context)
            system_prompt_len: System prompt length
            context_len: Context added per turn
            vocab_size: Vocabulary size

        Returns:
            prompts: List of token sequences (turn N's full context)
            turn_mapping: Mapping of prompt index to (session, turn)
        """
        # Generate system prompts (shared across all sessions)
        system_prompts = []
        for s in range(num_sessions):
            system = [s] + [randint(0, vocab_size) for _ in range(system_prompt_len - 1)]
            system_prompts.append(system)

        # Generate context chunks for each turn
        all_prompts = []
        turn_mapping = {}

        for s in range(num_sessions):
            session_context = list(system_prompts[s])

            for t in range(turns_per_session):
                # Turn content includes previous context (shared prefix!)
                turn_content = [t] + [randint(0, vocab_size) for _ in range(context_len - 1)]

                # Full prompt = system + all previous turns
                prompt = list(session_context) + turn_content
                all_prompts.append(prompt)
                turn_mapping[len(all_prompts) - 1] = (s, t)

                # Append to session context for next turn
                session_context.extend(turn_content)

        return all_prompts, turn_mapping

    def generate_batch_inference(
        self,
        num_requests: int = 64,
        system_prompt_len: int = 512,
        user_prompt_len: int = 256,
        num_unique_users: int = 8,
        vocab_size: int = 50000,
    ) -> tuple[List[List[int]], Dict[int, int]]:
        """
        Generate batch inference prompts with shared system prompts.

        Args:
            num_requests: Total number of requests
            system_prompt_len: System prompt length
            user_prompt_len: User prompt length (varies by user)
            num_unique_users: Number of unique users (each has own system prompt)
            vocab_size: Vocabulary size

        Returns:
            prompts: List of token sequences
            user_mapping: Mapping of prompt index to user index
        """
        # Generate user-specific system prompts
        system_prompts = []
        for u in range(num_unique_users):
            system = [u + 1000] + [randint(0, vocab_size) for _ in range(system_prompt_len - 1)]
            system_prompts.append(system)

        prompts = []
        user_mapping = {}

        for r in range(num_requests):
            user_id = r % num_unique_users
            user_mapping[r] = user_id

            # User-specific content
            user_content = [user_id + 2000] + [randint(0, vocab_size) for _ in range(user_prompt_len - 1)]

            # Full prompt = system + user content
            prompt = list(system_prompts[user_id]) + user_content
            prompts.append(prompt)

        return prompts, user_mapping


class CacheSimulator:
    """
    Simulates the KV cache behavior to estimate cache hit rates.

    This is a simplified simulation that doesn't require running actual inference,
    allowing us to measure the *potential* benefit of ContextPilot's reordering.
    """

    def __init__(self, block_size: int = 256):
        self.block_size = block_size
        self.hash_to_block = {}
        self.block_ref_counts = defaultdict(int)

    def compute_block_hash(self, token_ids: List[int], prefix_hash: int = -1) -> int:
        """Compute xxhash64 hash of a block."""
        import xxhash
        import numpy as np
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def simulate_allocation(
        self,
        prompts: List[List[int]],
    ) -> tuple[float, List[int]]:
        """
        Simulate KV cache allocation for a batch of prompts.

        Returns:
            cache_hit_rate: Fraction of tokens that were cache hits
            cached_token_counts: Tokens cached per prompt
        """
        self.hash_to_block = {}
        self.block_ref_counts = defaultdict(int)

        total_tokens = 0
        cached_tokens = 0
        cached_token_counts = []

        for prompt in prompts:
            num_blocks = (len(prompt) + self.block_size - 1) // self.block_size
            prompt_cached = 0
            prev_hash = -1

            for b in range(num_blocks):
                start = b * self.block_size
                end = min(start + self.block_size, len(prompt))
                block_tokens = prompt[start:end]

                # Compute hash with prefix chaining (like nano-vllm's BlockManager)
                block_hash = self.compute_block_hash(block_tokens, prev_hash)

                if block_hash in self.hash_to_block:
                    cached_tokens += len(block_tokens)
                    prompt_cached += len(block_tokens)
                    self.block_ref_counts[block_hash] += 1
                else:
                    self.hash_to_block[block_hash] = True

                # For next block, use current block's hash as prefix
                prev_hash = block_hash

            total_tokens += len(prompt)
            cached_token_counts.append(prompt_cached)

        cache_hit_rate = cached_tokens / total_tokens if total_tokens > 0 else 0.0
        return cache_hit_rate, cached_token_counts


def run_scenario(
    scenario_name: str,
    generator_fn: Callable,
    generator_kwargs: Dict[str, Any],
    use_context_optimizer: bool,
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    # Import ContextOptimizer directly to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "context_optimizer",
        os.path.join(os.path.dirname(__file__), "nanovllm", "context_optimizer.py")
    )
    context_optimizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_optimizer_module)
    ContextOptimizer = context_optimizer_module.ContextOptimizer

    # Generate prompts
    prompts, metadata = generator_fn(**generator_kwargs)

    if use_context_optimizer:
        # Measure optimization overhead
        t0 = time.time()
        optimizer = ContextOptimizer()
        reordered_prompts, execution_order = optimizer.reorder(prompts)
        optimization_overhead = time.time() - t0
        test_prompts = reordered_prompts
    else:
        optimization_overhead = 0.0
        test_prompts = prompts

    # Simulate cache allocation
    simulator = CacheSimulator()

    t0 = time.time()
    cache_hit_rate, _ = simulator.simulate_allocation(test_prompts)
    elapsed = time.time() - t0

    total_tokens = sum(len(p) for p in test_prompts)

    # For fair comparison, add optimization overhead to the "with" time
    total_time = elapsed + (optimization_overhead if use_context_optimizer else 0.0)

    return BenchmarkResult(
        scenario=scenario_name,
        num_prompts=len(test_prompts),
        total_tokens=total_tokens,
        without_optimization_time=elapsed if not use_context_optimizer else 0.0,
        with_optimization_time=total_time if use_context_optimizer else 0.0,
        without_cache_hit_rate=0.0 if use_context_optimizer else cache_hit_rate,
        with_cache_hit_rate=cache_hit_rate if use_context_optimizer else 0.0,
        speedup=0.0,
        optimization_overhead=optimization_overhead,
    )


def compare_scenarios(scenarios: List[str]):
    """Compare with and without ContextPilot optimization for each scenario."""
    # Import ContextOptimizer directly to avoid torch dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "context_optimizer",
        os.path.join(os.path.dirname(__file__), "nanovllm", "context_optimizer.py")
    )
    context_optimizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_optimizer_module)
    ContextOptimizer = context_optimizer_module.ContextOptimizer

    print("\n" + "=" * 70)
    print("  nano-vllm + ContextPilot Realistic Benchmark")
    print("=" * 70)

    # Scenario configurations
    scenario_configs = {
        "rag": {
            "name": "RAG Retrieval",
            "generator_fn": RealisticPromptGenerator().generate_rag_prompts,
            "kwargs": {
                "num_queries": 32,
                "chunks_per_query": 8,
                "chunk_size": 128,
                "num_shared_chunks": 4,
            },
            "description": "32 queries, each retrieving 8 chunks (4 shared)",
        },
        "multi_turn": {
            "name": "Multi-turn Chat",
            "generator_fn": RealisticPromptGenerator().generate_multi_turn_chat,
            "kwargs": {
                "num_sessions": 16,
                "turns_per_session": 8,
                "system_prompt_len": 256,
                "context_len": 512,
            },
            "description": "16 sessions x 8 turns, shared system + history",
        },
        "batch": {
            "name": "Batch Inference",
            "generator_fn": RealisticPromptGenerator().generate_batch_inference,
            "kwargs": {
                "num_requests": 64,
                "system_prompt_len": 512,
                "user_prompt_len": 256,
                "num_unique_users": 8,
            },
            "description": "64 requests, 8 unique users, shared system prompts",
        },
    }

    all_results = []

    for scenario in scenarios:
        if scenario not in scenario_configs:
            print(f"Unknown scenario: {scenario}")
            continue

        config = scenario_configs[scenario]
        print(f"\n{'-' * 70}")
        print(f"Scenario: {config['name']}")
        print(f"  {config['description']}")

        # Generate prompts ONCE
        print("  Generating prompts...", end=" ", flush=True)
        prompts, _ = config["generator_fn"](**config["kwargs"])

        # Shuffle prompts to simulate realistic out-of-order arrival
        import random
        random.seed(42)  # For reproducibility
        shuffled_prompts = list(prompts)
        random.shuffle(shuffled_prompts)
        print("done")

        # Simulate WITHOUT optimization (on shuffled prompts - worst case)
        print("  Running WITHOUT ContextPilot (shuffled prompts)...", end=" ", flush=True)
        simulator_naive = CacheSimulator()
        naive_hit_rate, _ = simulator_naive.simulate_allocation(shuffled_prompts)
        print("done")

        # Reorder with ContextPilot
        print("  Running WITH ContextPilot...", end=" ", flush=True)
        t0 = time.time()
        optimizer = ContextOptimizer()
        reordered, execution_order = optimizer.reorder(list(shuffled_prompts))
        optimization_time = time.time() - t0
        print("done")

        # Simulate WITH optimization (on reordered prompts)
        simulator_opt = CacheSimulator()
        opt_hit_rate, _ = simulator_opt.simulate_allocation(reordered)

        # Calculate results
        cache_improvement = opt_hit_rate - naive_hit_rate
        # Estimate speedup: 50% of cache improvement translates to prefill speedup
        estimated_speedup = 1.0 + max(0, cache_improvement) * 0.5

        print(f"\n  Results:")
        print(f"    Cache Hit Rate (naive/shuffled): {naive_hit_rate:.1%}")
        print(f"    Cache Hit Rate (optimized):     {opt_hit_rate:.1%}")
        print(f"    Cache Improvement:              {cache_improvement:+.1%}")
        print(f"    Optimization overhead:           {optimization_time*1000:.2f}ms")
        print(f"    Estimated Speedup:               {estimated_speedup:.2f}x")

        result_without = BenchmarkResult(
            scenario=config["name"],
            num_prompts=len(shuffled_prompts),
            total_tokens=sum(len(p) for p in shuffled_prompts),
            without_optimization_time=0.0,
            with_optimization_time=optimization_time,
            without_cache_hit_rate=naive_hit_rate,
            with_cache_hit_rate=opt_hit_rate,
            speedup=estimated_speedup,
            optimization_overhead=optimization_time,
        )

        result_with = result_without

        all_results.append((result_without, result_with))

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  {'Scenario':<25} {'Cache Before':<15} {'Cache After':<15} {'Speedup':<10}")
    print("-" * 70)
    for result_without, result_with in all_results:
        print(f"  {result_without.scenario:<25} {result_without.without_cache_hit_rate:<15.1%} {result_with.with_cache_hit_rate:<15.1%} {result_with.speedup:<10.2f}x")


def main():
    parser = argparse.ArgumentParser(description="nano-vllm + ContextPilot Realistic Benchmark")
    parser.add_argument(
        "--scenario",
        "-s",
        choices=["rag", "multi_turn", "batch", "all"],
        default="all",
        help="Benchmark scenario to run",
    )
    args = parser.parse_args()

    if args.scenario == "all":
        scenarios = ["rag", "multi_turn", "batch"]
    else:
        scenarios = [args.scenario]

    compare_scenarios(scenarios)


if __name__ == "__main__":
    main()
