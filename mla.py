# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import CacheConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.multi_stream_utils import execute_in_parallel
from vllm.utils.torch_utils import _encode_layer_name, aux_stream

logger = init_logger(__name__)

_SPARSE_MLA_BACKENDS_AUTO_FANOUT = ("FLASHMLA_SPARSE", "B12X_MLA_SPARSE")
_B12X_PREFILL_FASTPATH_MAX_Q = int(os.getenv("VLLM_B12X_INDEXER_EXTEND_MAX_Q", "8192"))


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse

        # Whether to skip top-k token selection computation in this layer.
        # When True, the indexer will not be called, and the layer will reuse
        # the topk_tokens buffer written by a previous layer in the same pass.
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = skip_topk
        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

        explicit_fanout = os.getenv("VLLM_ENABLE_MLA_PREATTN_FANOUT")
        if explicit_fanout is not None:
            fanout_enabled = envs.VLLM_ENABLE_MLA_PREATTN_FANOUT
        else:
            backend_name = self.mla_attn.attn_backend.get_name()
            fanout_enabled = backend_name in _SPARSE_MLA_BACKENDS_AUTO_FANOUT
            if fanout_enabled:
                logger.info_once(
                    "MLA pre-attn fan-out auto-enabled for %s backend "
                    "(set VLLM_ENABLE_MLA_PREATTN_FANOUT=0 to disable)",
                    backend_name,
                )

        self._mla_preattn_fanout = (
            fanout_enabled
            and self.indexer is not None
            and self.is_sparse
            and self.q_lora_rank is not None
            and self.fused_qkv_a_proj is not None
            and hasattr(self.indexer, "wk_weights_proj")
        )
        self._mla_midattn_fanout = self._mla_preattn_fanout and hasattr(
            self.indexer, "wq_b"
        )
        self._mla_dense_midattn_fanout = (
            envs.BOB_ENABLE_EAGLE3_FANOUT
            and self.q_lora_rank is not None
            and not self.is_sparse
        )

        if self._mla_preattn_fanout or self._mla_dense_midattn_fanout:
            aux_stream()
        if self._mla_preattn_fanout:
            self._fanout_start_event = torch.cuda.Event()
            self._fanout_done_event = torch.cuda.Event()
        else:
            self._fanout_start_event = None
            self._fanout_done_event = None
        if self._mla_midattn_fanout:
            self._fanout2_start_event = torch.cuda.Event()
            self._fanout2_done_event = torch.cuda.Event()
        else:
            self._fanout2_start_event = None
            self._fanout2_done_event = None
        if self._mla_dense_midattn_fanout:
            self._fanout_dense_start_event = torch.cuda.Event()
            self._fanout_dense_done_event = torch.cuda.Event()
        else:
            self._fanout_dense_start_event = None
            self._fanout_dense_done_event = None

    def _should_use_chunked_sparse_prefill_fast_path(
        self,
        hidden_states: torch.Tensor,
    ) -> bool:
        if (
            self.indexer is None
            or not self.is_sparse
            or self.skip_topk
            or not self.mla_attn.use_direct_call
            or self.topk_indices_buffer is None
        ):
            return False
        if self.mla_attn.attn_backend.get_name() != "B12X_MLA_SPARSE":
            return False
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.mla_attn.layer_name)
            indexer_metadata = attn_metadata_raw.get(self.indexer.k_cache.prefix)
        elif isinstance(attn_metadata_raw, list):
            return False
        else:
            return False
        if attn_metadata is None or indexer_metadata is None:
            return False
        prefill = getattr(indexer_metadata, "prefill", None)
        chunks = getattr(prefill, "chunks", None)
        if not chunks or len(chunks) <= 1:
            return False
        if getattr(attn_metadata, "num_reqs", 0) != 1:
            return False
        if getattr(attn_metadata, "max_query_len", 0) <= 1:
            return False
        if (
            getattr(attn_metadata, "num_actual_tokens", hidden_states.shape[0])
            <= _B12X_PREFILL_FASTPATH_MAX_Q
        ):
            return False
        if getattr(indexer_metadata, "num_decodes", 0) != 0:
            return False
        return True

    @staticmethod
    def _slice_b12x_attn_metadata(attn_metadata, token_start: int, token_end: int):
        row_count = max(0, token_end - token_start)
        return type(attn_metadata)(
            num_reqs=attn_metadata.num_reqs,
            max_query_len=row_count,
            max_seq_len=attn_metadata.max_seq_len,
            num_actual_tokens=row_count,
            req_id_per_token=attn_metadata.req_id_per_token[token_start:token_end],
            cache_seq_lens_per_req=attn_metadata.cache_seq_lens_per_req,
            cache_seq_lens_per_token=attn_metadata.cache_seq_lens_per_token[
                token_start:token_end
            ],
            block_table=attn_metadata.block_table,
            page_table_1=attn_metadata.page_table_1[token_start:token_end],
            nsa_cache_seqlens=attn_metadata.nsa_cache_seqlens[token_start:token_end],
            nsa_cu_seqlens=attn_metadata.nsa_cu_seqlens[: row_count + 1],
            nsa_cu_seqlens_k=attn_metadata.nsa_cu_seqlens_k[: row_count + 1],
            block_size=attn_metadata.block_size,
            topk_tokens=attn_metadata.topk_tokens,
            physical_token_table=attn_metadata.physical_token_table,
            physical_token_table_width=attn_metadata.physical_token_table_width,
        )

    def _forward_chunked_sparse_prefill_fast_path(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        indexer_kw: torch.Tensor | None,
        indexer_q_raw: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.mla_attn.layer_name]
        indexer_metadata = attn_metadata_raw[self.indexer.k_cache.prefix]
        prefill = indexer_metadata.prefill
        assert prefill is not None

        q_fp8, indexer_k, weights = self.indexer.prepare_sparse_inputs(
            hidden_states,
            q_c,
            positions,
            self.indexer_rope_emb,
            kw=indexer_kw,
            q_raw=indexer_q_raw,
        )
        self.indexer.cache_indexer_k(indexer_k, indexer_metadata.slot_mapping)
        if self.mla_attn.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                q,
                kv_c_normed,
                k_pe,
                _encode_layer_name(self.mla_attn.layer_name),
            )

        if llama_4_scaling is not None:
            q = q * llama_4_scaling

        output = torch.empty(
            (hidden_states.shape[0], self.num_heads * self.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        kv_cache = self.mla_attn.kv_cache
        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        layer_slot_mapping = slot_mapping.get(self.mla_attn.layer_name)
        if layer_slot_mapping is None:
            raise RuntimeError(
                f"Missing slot mapping for MLA layer {self.mla_attn.layer_name}"
            )

        for chunk in prefill.chunks:
            token_start = int(chunk.token_start)
            token_end = int(chunk.token_end)
            attn_chunk = self._slice_b12x_attn_metadata(
                attn_metadata,
                token_start,
                token_end,
            )
            self.indexer.run_prefill_chunk(hidden_states, q_fp8, weights, chunk)
            self.mla_attn.impl.do_kv_cache_update(  # type: ignore[attr-defined]
                kv_c_normed[token_start:token_end],
                k_pe[token_start:token_end],
                kv_cache,
                layer_slot_mapping[token_start:token_end],
                self.mla_attn.kv_cache_dtype,
                self.mla_attn._k_scale,
            )
            chunk_q = q[token_start:token_end]
            chunk_output = output[token_start:token_end]
            self.mla_attn.forward_impl(
                chunk_q,
                kv_c_normed[token_start:token_end],
                k_pe[token_start:token_end],
                kv_cache,
                attn_chunk,
                output=chunk_output,
                sparse_token_start=token_start,
            )
        return self.o_proj(output)[0]

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None
        indexer_kw: torch.Tensor | None = None
        indexer_q_raw: torch.Tensor | None = None
        kv_c_normed_pre: torch.Tensor | None = None
        k_pe_pre: torch.Tensor | None = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )

            if self._mla_preattn_fanout:
                indexer = self.indexer
                qkv_lora, [indexer_kw] = execute_in_parallel(
                    default_fn=lambda: self.fused_qkv_a_proj(hidden_states)[0],
                    aux_fns=[lambda: indexer.wk_weights_proj(hidden_states)[0]],
                    start_event=self._fanout_start_event,
                    done_events=[self._fanout_done_event],
                    aux_streams=lambda: [aux_stream()],
                    enable=True,
                )
            else:
                qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]

            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)

            if self._mla_midattn_fanout:
                indexer = self.indexer
                indexer_q_raw, [q] = execute_in_parallel(
                    default_fn=lambda: indexer.wq_b(q_c)[0],
                    aux_fns=[lambda: self.q_b_proj(q_c)[0]],
                    start_event=self._fanout2_start_event,
                    done_events=[self._fanout2_done_event],
                    aux_streams=lambda: [aux_stream()],
                    enable=True,
                )
            elif self._mla_dense_midattn_fanout and aux_stream() is not None:
                kv_lora_local = kv_lora

                def kv_path():
                    kv_c_loc, k_pe_loc = kv_lora_local.split(
                        [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                    )
                    return self.kv_a_layernorm(kv_c_loc), k_pe_loc

                q, [(kv_c_normed_pre, k_pe_pre)] = execute_in_parallel(
                    default_fn=lambda: self.q_b_proj(q_c)[0],
                    aux_fns=[kv_path],
                    start_event=self._fanout_dense_start_event,
                    done_events=[self._fanout_dense_done_event],
                    aux_streams=lambda: [aux_stream()],
                    enable=True,
                )
            else:
                q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        if kv_c_normed_pre is not None:
            kv_c_normed = kv_c_normed_pre
            k_pe = k_pe_pre
        else:
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim :], k_pe
            )

        if self.indexer and self.is_sparse and not self.skip_topk:
            if self._should_use_chunked_sparse_prefill_fast_path(hidden_states):
                return self._forward_chunked_sparse_prefill_fast_path(
                    positions=positions,
                    hidden_states=hidden_states,
                    q_c=q_c,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    indexer_kw=indexer_kw,
                    indexer_q_raw=indexer_q_raw,
                    llama_4_scaling=llama_4_scaling,
                )
            self.indexer(
                hidden_states,
                q_c,
                positions,
                self.indexer_rope_emb,
                kw=indexer_kw,
                q_raw=indexer_q_raw,
            )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        attn_out = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
        )

        return self.o_proj(attn_out)[0]
