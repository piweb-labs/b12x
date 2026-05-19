B12X_MLA_SPARSE prefill optimization pack

Contents:
- vllm/v1/attention/backends/mla/b12x_mla_sparse.py
- vllm/v1/attention/backends/mla/indexer.py
- vllm/model_executor/layers/sparse_attn_indexer.py
- b12x_prefill_fastpack.patch

Purpose:
- reduce B12X_MLA_SPARSE prefill overhead
- replace dense-logits-based prefill chunking with B12X tiled-topk aware chunking
- keep dense fallback safe via internal subchunking
- reduce per-layer logical->physical metadata overhead in b12x_mla_sparse

Apply options:
1. overlay these files into repo root
2. or from repo root: git apply b12x_prefill_fastpack.patch

Verified:
- python3 -m py_compile on the 3 Python files
