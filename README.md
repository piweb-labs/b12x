# b12x — Optimized vLLM Patches for NSA / MLA on PCIe GPUs

Patches for [vLLM b12x](https://github.com/lukealonso/b12x) focused on **prefill throughput** and **NSA correctness** on multi-GPU PCIe setups (no NVLink).

Tested on 8× RTX PRO 6000 Blackwell (768 GB VRAM), TP=8, running GLM-5.1 and Kimi-K2.6.

---

## What's Inside

### 1. Prefill Fastpack — Chunk-Pipelined Sparse Attention

**Files:** `b12x_mla_sparse.py`, `indexer.py`, `sparse_attn_indexer.py`, `mla.py`, `mla_attention.py`, `deepseek_v2.py`

Rewrites the prefill execution from **stage-serial** to **chunk-pipelined**:

| | Original | b12x |
|---|---|---|
| Index | Run entire indexer → barrier | Chunk i index → immediately consume |
| KV update | After all indexing done | Chunk i index → chunk i cache update |
| Attention | After all cache updates | Chunk i cache → chunk i attention |
| Overlap | None (3 global barriers) | Per-chunk pipeline, minimal gaps |

### 2. GLM-5.1 NSA+MTP Tool-Call XML Leak Fix

**Files:** `glm51_nsa_mtp_tool_parser.py`, `sitecustomize.py`

Drop-in vLLM tool-parser plugin that patches 7 failure modes of tool-call corruption under NSA + MTP speculative decoding:

1. Markdown fence around whole args ` ```json\n{...}\n``` `
2. Markdown fence around individual values
3. Stray `\n` inside JSON string values
4. Duplicate `arguments` key in tool call
5. Truncated JSON (missing closing `}`)
6. Interleaved plain-text leakage from NSA window
7. MTP draft token leaking raw XML `<invoke>` tags

The `sitecustomize.py` hook monkey-patches vLLM's `build_logitsprocs` to allow custom logits processors to coexist with speculative decoding (MTP).

---

## Key Insights from This Optimization

### 1. The bottleneck isn't the attention kernel — it's the execution organization

The original B12X_MLA_SPARSE attention kernel itself is fast. What kills prefill throughput is how it's **orchestrated**:

```
Original:  [entire indexer] → barrier → [entire kv update] → barrier → [entire attention]
```

This turns long-context prefill into a hard serial barrier chain. NSA's core advantage — "only fetch a sparse subset" — gets eaten by scheduling overhead.

### 2. The problem isn't the sparse principle — it's the implementation violating it

NSA should convert "the top-k for this small query segment" into consumable data **as fast as possible**, then immediately proceed to attention computation. The original implementation prepares the **entire index first** — effectively paying the full traversal tax before any attention starts.

**The implementation negated the algorithm's design intent.**

### 3. Under hard matrix-size constraints, pipeline your chunks — don't enlarge them

Given `extend_max_q <= 8192` (no oversized matrices), you can't speed things up by "throwing a bigger Q at it in one shot." The right approach is:

```
Chunk i: index → cache update → attention → next chunk
```

Not: wait for all chunks' indexing to finish before doing anything.

### 4. The key to long-context optimization is reducing GPU idle time between stages, not reducing Python calls

The big win came from replacing **stage-level barriers** with **chunk-level continuous consumption**. This is harder than tweaking topk logic or traversal patterns — but the payoff is far more substantial.

**Eliminate global waits. Pipeline everything.**

### 5. Long contexts will always be chunked — and that's fine

128K / 200K sequences will continue to be split into multiple chunks. This isn't a bug — it's a consequence of the power/VRAM boundary (`extend_max_q <= 8192`). The question was never "can we avoid chunking?" but rather:

- **Per-chunk index construction cost** — make it cheaper
- **Inter-chunk launch/sync gaps** — make them smaller  
- **Indexer → attention handoff overhead** — make it tighter

### 6. Chunk-pipelining is the general principle, NSA is just the first application

The same pattern applies anywhere you have: sparse selection → data preparation → computation. If you can make the selection produce consumable output early, pipeline it. Don't batch-prepare then batch-compute.

---

## Requirements

- vLLM b12x branch (lukealonso/b12x)
- FlashInfer
- 8× GPU with PCIe Gen5 x16 (tested on RTX PRO 6000 Blackwell)
- CUDA 12.6+

## Usage

```bash
# Apply prefill fastpack patches
cd /path/to/vllm
git apply b12x_prefill_fastpack.patch

# Or replace individual files manually:
# vllm/v1/attention/backends/mla/b12x_mla_sparse.py
# vllm/v1/attention/backends/mla/indexer.py
# vllm/model_executor/layers/sparse_attn_indexer.py
# vllm/model_executor/layers/mla.py
# vllm/model_executor/layers/attention/mla_attention.py
# vllm/model_executor/models/deepseek_v2.py

# For GLM-5.1 tool-call fix, add to vLLM plugins:
export PYTHONPATH="/path/to/plugins:$PYTHONPATH"
export VLLM_TOOL_PARSER_PLUGIN=glm51_nsa_mtp_tool_parser
```

## Acknowledgments

- [Luke Alonso](https://github.com/lukealonso) — original b12x branch, vLLM PCIe optimizations
- [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) — RTX 6000 Pro deployment wiki
- DeepSeek team — NSA (Native Sparse Attention) algorithm
- vLLM project

## License

Apache-2.0 (same as vLLM)
