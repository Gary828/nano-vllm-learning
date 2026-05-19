(nano) root@DY-20230804LNIS:~/study/nano-vllm# python bench_kv_quant.py 

========================================================================
KV Cache Quant Benchmark
========================================================================
Prompt length: 4096 tokens
Max output tokens: 32

Mode       | Time(s) | TTFT(s) | Init GB | Gen Peak GB | Runtime +GB | KV Blocks | Est. Max Live Seqs
---------- | ------- | ------- | ------- | ----------- | ----------- | --------- | ------------------
baseline   |  31.363 |  14.780 |   7.317 |      10.080 |       2.763 |       226 |                 13
int8       |  55.534 |  23.836 |   6.228 |      10.569 |       4.342 |       361 |                 21
fp8_e4m3fn |  60.520 |  26.910 |   6.210 |      10.749 |       4.539 |       371 |                 21

int8 KV block gain: 1.60x
int8 estimated live-sequence gain: 1.62x
int8 init allocated delta: -1.090 GB
int8 generate peak delta: +0.489 GB
int8 runtime temporary delta: +1.579 GB
fp8_e4m3fn KV block gain: 1.64x
fp8_e4m3fn estimated live-sequence gain: 1.62x
fp8_e4m3fn init allocated delta: -1.107 GB
fp8_e4m3fn generate peak delta: +0.669 GB
fp8_e4m3fn runtime temporary delta: +1.776 GB
(nano) root@DY-20230804LNIS:~/study/nano-vllm# python eval_kv_quant_quality.py 

========================================================================
KV Cache Quant Quality Check
========================================================================
Exact output match rate: 0/6 = 0.00%

[1] diff
  baseline: Also, explain why it's important for the model to have a large enough number of parameters in the model. Additionally,
  quant   : udos<|endoftext|>aling according sake differently the seenjącym iils the skilled and and equivalent ab viewing the Here equivalent within within within
[2] diff
  baseline: Additionally, explain why each of these methods is effective.
Answer:

The goal is to reduce GPU memory usage during LLM
  quant   : S Sooo. from.stS S Ppa total por pref #0 ordered S ..SpendIP kIMID
[3] diff
  baseline: Also, explain the difference between prefill and decode in three sentences. Additionally, explain the difference between prefill and decode
  quant   : v one corresponding plus without plus without: view view view viewviewviewvalues as green ; order, the same
[4] diff
  baseline: The secret passkey is 60494.
Repeat the secret passkey only. The secret passkey is
  quant   : 6666 up up up up up up up up up66666666_tpl up up
[5] diff
  baseline: The secret passkey is 65125.
Repeat the secret passkey only. The secret passkey is
  quant   : 6666666666666666666ment666ment
[6] diff
  baseline: Answer:
15306

The secret passkey is 15306.
Answer:
1
  quant   : mentmentmentoneyoneymayıoneyoneymayımentoneymayıment tjmentoneyoneymayımentmentoney hundredmentment



nano) root@DY-20230804LNIS:~/study/nano-vllm# python eval_kv_quant_quality.py --quant-mode fp8_e4m3fn --no-enforce-eager
args:

Namespace(model_path='/root/study/lite_llama/my_weight/qwen3-0.6B', max_tokens=24, quant_mode='fp8_e4m3fn', enforce_eager=False)

========================================================================
KV Cache Quant Quality Check
========================================================================
Exact output match rate: 0/6 = 0.00%

[1] diff
  baseline: Also, explain why it's important for the model to have a large enough number of parameters in the model. Additionally,
  quant   : Reader SQichte softj esteemed Mling w (yyitoryalthnacha textervers not wimim thugg
[2] diff
  baseline: Additionally, explain why each of these methods is effective.
Answer:

The goal is to reduce GPU memory usage during LLM
  quant   : of Plussequ. outside step from s. part H modern step content from best best best system mail from best S best
[3] diff
  baseline: Also, explain the difference between prefill and decode in three sentences. Additionally, explain the difference between prefill and decode
  quant   : Participants as through to.putExtra peopleatif as step .atif put method put much pen pen pen pen pen and pen pen terms
[4] diff
  baseline: The secret passkey is 60494.
Repeat the secret passkey only. The secret passkey is
  quant   : 9999ipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipation9ipation
[5] diff
  baseline: The secret passkey is 65125.
Repeat the secret passkey only. The secret passkey is
  quant   : 556ipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipation
[6] diff
  baseline: Answer:
15306

The secret passkey is 15306.
Answer:
1
  quant   : 55 tin missing missing missing missingipationipationipation missing missing missing missing missing missingipation missingipation missingipation missing missing missing



(nano) root@DY-20230804LNIS:~/study/nano-vllm# python eval_kv_quant_quality.py --quant-mode fp8_e4m3fn
args:

Namespace(model_path='/root/study/lite_llama/my_weight/qwen3-0.6B', max_tokens=24, quant_mode='fp8_e4m3fn', enforce_eager=True)

========================================================================
KV Cache Quant Quality Check
========================================================================
Exact output match rate: 0/6 = 0.00%

[1] diff
  baseline: Also, explain why it's important for the model to have a large enough number of parameters in the model. Additionally,
  quant   : Reader SQichte softj esteemed Mling w (yyitoryalthnacha textervers not wimim thugg
[2] diff
  baseline: Additionally, explain why each of these methods is effective.
Answer:

The goal is to reduce GPU memory usage during LLM
  quant   : of Plussequ. outside step from s. part H modern step content from best best best system mail from best S best
[3] diff
  baseline: Also, explain the difference between prefill and decode in three sentences. Additionally, explain the difference between prefill and decode
  quant   : Participants as through to.putExtra peopleatif as step .atif put method put much pen pen pen pen pen and pen pen terms
[4] diff
  baseline: The secret passkey is 60494.
Repeat the secret passkey only. The secret passkey is
  quant   : 9999ipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipation9ipation
[5] diff
  baseline: The secret passkey is 65125.
Repeat the secret passkey only. The secret passkey is
  quant   : 556ipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipationipation
[6] diff
  baseline: Answer:
15306

The secret passkey is 15306.
Answer:
1
  quant   : 55 tin missing missing missing missingipationipationipation missing missing missing missing missing missingipation missingipation missingipation missing missing missing

  

  (nano) root@DY-20230804LNIS:~/study/nano-vllm# pytest -q tests/test_kv_quant.py tests/test_attention_kv_cache_staleness.py
............                                                                                                                                   [100%]
12 passed in 1.71s