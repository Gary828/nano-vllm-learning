import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        eos = self.tokenizer.eos_token_id
        config.eos = set(eos) if isinstance(eos, list) else {eos}
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        use_context_optimizer: bool = True,
    ):
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        self.scheduler.cache_aware = use_context_optimizer

        request_seq_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            request_seq_ids.append(self.add_request(prompt, sp))

        outputs = {}
        prefill_throughput = 0.0
        decode_throughput = 0.0
        first_token_time = None
        first_decode_step_time = None
        per_request_ttft = {}
        per_request_completion_time = {}

        start_time = perf_counter()
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            elapsed = perf_counter() - start_time

            if first_token_time is None:
                has_finished_output = len(output) > 0
                has_running_completion = any(
                    seq.num_completion_tokens > 0 for seq in self.scheduler.running
                )
                if has_finished_output or has_running_completion:
                    first_token_time = elapsed

            for seq in self.scheduler.running:
                if seq.num_completion_tokens > 0 and seq.seq_id not in per_request_ttft:
                    per_request_ttft[seq.seq_id] = elapsed
            for seq_id, _ in output:
                if seq_id not in per_request_ttft:
                    per_request_ttft[seq_id] = elapsed

            if first_decode_step_time is None and num_tokens < 0:
                first_decode_step_time = elapsed

            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)

            pbar.set_postfix(
                {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
            )

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if seq_id not in per_request_completion_time:
                    per_request_completion_time[seq_id] = elapsed
                pbar.update(1)

        pbar.close()

        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        total_time = perf_counter() - start_time
        per_request_ttft = [per_request_ttft.get(seq_id) for seq_id in request_seq_ids]
        per_request_completion_time = [per_request_completion_time.get(seq_id) for seq_id in request_seq_ids]

        return {
            "outputs": outputs,
            "ttft": first_token_time,
            "ttft_token": first_token_time,
            "batch_first_token_time": first_token_time,
            "per_request_ttft": per_request_ttft,
            "per_request_completion_time": per_request_completion_time,
            "ttfd_decode_step": first_decode_step_time,
            "total_time": total_time,
        }
