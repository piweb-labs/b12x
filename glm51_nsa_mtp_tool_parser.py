"""GLM-5.1 NSA+MTP tool fix plugin.

Drop-in vLLM tool-parser plugin that patches tool-call corruption under NSA + MTP.

Failure modes patched at the PARSER layer (this file's main job):
  1. Markdown fence around whole args              ```json\n{...}\n```
  2. Markdown fence around individual values        "k": ```\n{...}\n```
  3. Bare backticks injected into JSON structure    "k": ```<content>```, ...
  4. Trailing commas                                {"a":1,"b":2,}
  5. Unescaped newlines/quotes in strings           {"text": "line1\nline2"}
  6. MTP mid-splice JSON corruption                 valid prefix + garbled tail
  7. Zero-arg streaming tool call silently dropped  <tool_call>name</tool_call>

Separately exported here (export `GlmToolEmissionGuard`): a per-request logits
processor that the SERVER OR CLIENT can attach to fix the model-side
"no_xml_emitted" bug (model emits stop_token without first opening `<tool_call>`,
typical under NSA + MTP). The processor masks GLM stop tokens until either a
`<tool_call>` has been opened OR a max-content-token budget has elapsed.

How to load PARSER:
    vllm serve ... --tool-parser-plugin /path/to/glm51_nsa_mtp_tool_parser.py \
                   --tool-call-parser glm51_nsa_mtp

How to attach LOGITS PROCESSOR (client side, per-request):
    body["extra_body"] = {"logits_processors": ["glm51_tool_emission_guard"]}
    # requires vllm to be started with this module discoverable via PYTHONPATH /
    # --logits-processor-pattern, or a server-side hook.

Tested on vLLM 0.20.2rc1.dev249+glmkimirebase20260514, b12x c929144,
GLM-5.1 NVFP4 with MTP num_speculative_tokens=3, B12X_MLA_SPARSE attention.
"""
from __future__ import annotations

import json
import re

try:
    import json_repair as _json_repair_mod
    HAVE_JSON_REPAIR = True
except ImportError:
    HAVE_JSON_REPAIR = False

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParserManager
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

logger = init_logger(__name__)


# ---------- correction primitives ----------

_MD_WHOLE = re.compile(
    r"^\s*```(?:json|javascript|python|js|py|jsonl)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_MD_LEADING_LABEL = re.compile(r"^(?:json|JSON|Json)\s*:?\s*\n?")
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_INLINE_FENCE = re.compile(
    r'(:\s*)"?```(?:json|javascript|python|js|py|jsonl)?n?\s*\n?(.*?)\n?```"?n?',
    re.DOTALL | re.IGNORECASE,
)


def _strip_orphan_backticks(s: str) -> str:
    out, in_str, esc = [], False, False
    for c in s:
        if esc:
            out.append(c); esc = False; continue
        if c == "\\" and in_str:
            out.append(c); esc = True; continue
        if c == '"':
            in_str = not in_str; out.append(c); continue
        if c == "`" and not in_str:
            continue
        out.append(c)
    return "".join(out)


def _preprocess_inline_fences(args_str: str) -> tuple[str, list[str]]:
    """Replace bare ```...``` blocks appearing as JSON values with proper JSON."""
    fixes = []
    def replace(m):
        prefix, inner = m.group(1), m.group(2)
        s = _TRAILING_COMMA.sub(r"\1", inner.strip())
        try:
            obj = json.loads(s)
            fixes.append("inline_fence_to_value")
            return f"{prefix}{json.dumps(obj, ensure_ascii=False)}"
        except json.JSONDecodeError:
            fixes.append("inline_fence_to_string")
            return f"{prefix}{json.dumps(inner, ensure_ascii=False)}"
    new_str, _ = _INLINE_FENCE.subn(replace, args_str)
    return new_str, fixes


def _try_json_repair(value: str) -> tuple[object | None, list[str]]:
    fixes = []
    candidates = [value]
    s = value.strip()
    if s != value: candidates.append(s)
    s2 = _MD_WHOLE.match(s).group(1).strip() if _MD_WHOLE.match(s) else s
    if s2 != s: candidates.append(s2)
    s3 = _MD_LEADING_LABEL.sub("", s2, count=1)
    if s3 != s2: candidates.append(s3)
    s4 = _TRAILING_COMMA.sub(r"\1", s3)
    if s4 != s3: candidates.append(s4)
    s5 = _strip_orphan_backticks(s4)
    if s5 != s4: candidates.append(_TRAILING_COMMA.sub(r"\1", s5))
    for i, cand in enumerate(candidates):
        try:
            obj = json.loads(cand)
            for stage in ("strip","strip_md","strip_label","trailing_comma","orphan_backticks")[:i]:
                fixes.append(stage)
            return obj, fixes
        except json.JSONDecodeError:
            continue
    if HAVE_JSON_REPAIR:
        try:
            obj = json.loads(_json_repair_mod.repair_json(value, skip_json_loads=False))
            return obj, fixes + ["json_repair"]
        except Exception:
            pass
    return None, fixes


def correct_args_json(args_str: str, tool_params: dict | None = None) -> tuple[str, list[str]]:
    """Return (corrected_args_json_str, list_of_fixes_applied). Always returns a
    string; if no correction needed/possible, returns args_str unchanged."""
    if not args_str or args_str == "{}":
        return args_str, []
    applied = []
    pre, pre_fixes = _preprocess_inline_fences(args_str)
    if pre_fixes:
        applied.extend(pre_fixes); args_str = pre
    try:
        obj = json.loads(args_str)
    except json.JSONDecodeError:
        repaired, fixes = _try_json_repair(args_str)
        if repaired is not None:
            applied.extend(f"full:{f}" for f in fixes)
            return json.dumps(repaired, ensure_ascii=False), applied
        return args_str, applied
    if not isinstance(obj, dict):
        return args_str, applied
    props = (tool_params or {}).get("properties", {}) if tool_params else {}
    changed = False
    for k, v in list(obj.items()):
        if isinstance(v, str):
            exp = props.get(k, {}).get("type")
            looks_jsony = v.strip().startswith(("{", "[", "```"))
            if looks_jsony and exp in ("object", "array", None):
                repaired, fixes = _try_json_repair(v)
                if repaired is not None and not isinstance(repaired, str):
                    obj[k] = repaired
                    applied.append(f"{k}:{','.join(fixes)}")
                    changed = True
    if changed:
        return json.dumps(obj, ensure_ascii=False), applied
    return args_str, applied


# ---------- the parser plugin ----------

@ToolParserManager.register_module("glm51_nsa_mtp")
class Glm51NsaMtpToolParser(Glm47MoeModelToolParser):
    """Buffering streaming parser: collects all tool-call XML during the stream,
    applies correction at completion, then emits ONE corrected delta with full
    args at the end. Sacrifices fine-grained streaming of arg chars for correctness
    under NSA+MTP corruption."""

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        # name -> parameters schema for arg type lookup
        self._tools_by_name: dict[str, dict] = {}
        for t in (tools or []):
            try:
                name = t.function.name
                params = t.function.parameters or {}
                self._tools_by_name[name] = params
            except Exception:
                pass
        # Tracks per-tool-index whether we've emitted name/args yet (override parent state)
        self._buffered_emitted_for: set[int] = set()
        # Counters for periodic logging
        self._stats = {"non_stream_calls": 0, "non_stream_fixes": 0,
                       "stream_emits": 0, "stream_fixes": 0,
                       "zero_arg_emits": 0}
        logger.info(
            "[glm51_nsa_mtp] parser INSTANTIATED  tools=%d (names=%s)  "
            "json_repair_available=%s",
            len(self._tools_by_name), list(self._tools_by_name.keys())[:8],
            HAVE_JSON_REPAIR,
        )

    def extract_tool_calls(self, model_output: str, request):
        """Non-streaming: run parent, then post-process tool args."""
        info = super().extract_tool_calls(model_output, request)
        if not getattr(info, "tools_called", False):
            return info
        self._stats["non_stream_calls"] += 1
        try:
            for tc in info.tool_calls or []:
                args_str = tc.function.arguments
                params = self._tools_by_name.get(tc.function.name)
                corrected, fixes = correct_args_json(args_str, params)
                if fixes:
                    self._stats["non_stream_fixes"] += 1
                    logger.info(
                        "[glm51_nsa_mtp] FIX non-stream tool=%s fixes=%s  "
                        "in_len=%d out_len=%d  totals=%s",
                        tc.function.name, fixes, len(args_str), len(corrected),
                        self._stats,
                    )
                    tc.function.arguments = corrected
        except Exception as e:
            logger.exception("[glm51_nsa_mtp] non-stream correction failed: %s", e)
        return info

    def extract_tool_calls_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids, request,
    ):
        """Streaming: BUFFER all tool calls; emit nothing for tool deltas until the
        entire tool_call region is closed (i.e. we see </tool_call>). At that point
        emit name + corrected full args in one DeltaMessage. Non-tool-call content
        (reasoning, regular content) still streams through normally."""
        if not self._tools_enabled(request):
            return DeltaMessage(content=delta_text) if delta_text else None

        content = self._extract_content(current_text)
        regions = self._extract_tool_call_regions(current_text)
        deltas: list[DeltaToolCall] = []

        for i, (inner_text, is_complete) in enumerate(regions):
            self._ensure_tool_state_for(i)

            # Extract tool name even for zero-arg complete region (fix bug #7)
            tool_name = self._extract_tool_name_from_region(inner_text)
            if tool_name is None and is_complete:
                # GLM-4.7 zero-arg complete: <tool_call>name</tool_call>
                candidate = inner_text.strip()
                if candidate:
                    tool_name = candidate
            if not tool_name:
                continue  # name not yet complete

            if i in self._buffered_emitted_for:
                continue  # already emitted

            if not is_complete:
                continue  # wait until </tool_call> seen

            # Region complete — build args from XML pairs, correct, emit once
            args_so_far = self._build_args_json_so_far(tool_name, inner_text, True)
            if not args_so_far:
                args_so_far = "{}"

            params = self._tools_by_name.get(tool_name)
            corrected, fixes = correct_args_json(args_so_far, params)
            self._stats["stream_emits"] += 1
            is_zero_arg = (args_so_far in ("{}", "")) or (corrected in ("{}", ""))
            if is_zero_arg:
                self._stats["zero_arg_emits"] += 1
            if fixes:
                self._stats["stream_fixes"] += 1
                logger.info(
                    "[glm51_nsa_mtp] FIX stream tool[%d]=%s fixes=%s  "
                    "in_len=%d out_len=%d  totals=%s",
                    i, tool_name, fixes, len(args_so_far), len(corrected),
                    self._stats,
                )
            elif is_zero_arg:
                logger.debug(
                    "[glm51_nsa_mtp] EMIT stream tool[%d]=%s zero-arg (parent would drop)",
                    i, tool_name,
                )

            # Sync parent state so any subsequent re-entries see we're done
            self.prev_tool_call_arr[i]["name"] = tool_name
            self.prev_tool_call_arr[i]["arguments"] = corrected
            self.streamed_args_for_tool[i] = corrected
            self._buffered_emitted_for.add(i)

            deltas.append(
                DeltaToolCall(
                    index=i,
                    id=self._tool_call_ids[i],
                    type="function",
                    function=DeltaFunctionCall(
                        name=tool_name,
                        arguments=corrected,
                    ).model_dump(exclude_none=True),
                )
            )

        if regions:
            self.current_tool_id = len(regions) - 1

        if content or deltas:
            return DeltaMessage(content=content, tool_calls=deltas)
        return None


# =========================================================================
# Logits processor: guard against premature stop-token emission ("no_xml")
# =========================================================================
#
# vLLM v1 LogitsProcessor. After the model emits </think> (154842), the
# next non-content token in a tool-armed context must be either
# `<tool_call>` (154843) or normal content followed eventually by a stop.
# Under NSA+MTP, GLM-5.1 sometimes emits a stop token directly after
# </think> without first opening `<tool_call>`, producing
# `finish_reason=tool_calls` with zero XML.
#
# Fix: for the first `min_tokens_after_think` decode steps following
# </think>, mask the GLM stop tokens UNLESS `<tool_call>` has already
# opened. Plain-text answers are unaffected once the small budget passes.
#
# Token IDs (probed from /mnt/wsl-vllm/KINZE-GLM-5.1 tokenizer):
#   154820 <|endoftext|>   eos / pad
#   154827 <|user|>        turn boundary
#   154828 <|assistant|>
#   154829 <|observation|> tool-result boundary
#   154841 <think>
#   154842 </think>
#   154843 <tool_call>
#   154844 </tool_call>
#
# Register via vLLM:
#   vllm serve ... \
#     --logits-processors glm51_nsa_mtp_tool_parser:GlmToolEmissionGuard
#
# Tunable via env:
#   GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK   default 16 (decode steps)
#   GLM_TOOL_GUARD_DISABLE                  set to 1 to no-op

import os
import sys as _sys

import torch

# Print BEFORE doing any vllm imports — confirms module body is being executed
print(
    f"[glm51_nsa_mtp.MODULE-EXEC] pid={os.getpid()} __name__={__name__} "
    f"__file__={__file__} cached={'glm51_nsa_mtp_tool_parser' in _sys.modules}",
    file=_sys.stderr, flush=True,
)

# HARD requirement: vllm v1 enforces `issubclass(cls, LogitsProcessor)`.
# No try/except — if this import fails, fail loudly with the real traceback.
from vllm.v1.sample.logits_processor.interface import (  # type: ignore
    LogitsProcessor as _LogitsProcessorBase,
)
print(
    f"[glm51_nsa_mtp.import] OK pid={os.getpid()} base={_LogitsProcessorBase} "
    f"base_id={id(_LogitsProcessorBase)}",
    file=_sys.stderr, flush=True,
)


GLM_STOP_IDS: tuple[int, ...] = (154820, 154827, 154829)
GLM_THINK_END_ID: int = 154842
GLM_TOOL_CALL_OPEN_ID: int = 154843


class GlmToolEmissionGuard(_LogitsProcessorBase):
    """vLLM v1 LogitsProcessor — masks GLM stop tokens for a short window
    after </think> until <tool_call> opens. Self-disengages after the
    budget elapses, so plain-text answers are not blocked.

    Inherits from `vllm.v1.sample.logits_processor.interface.LogitsProcessor`
    (or a stub if v1 isn't installed). vLLM enforces
    `issubclass(cls, LogitsProcessor)` on classes passed via
    `--logits-processors`, so inheritance is required.
    """

    def __init__(self, vllm_config, device, is_pin_memory) -> None:
        self.device = device
        self.pin_memory = is_pin_memory
        self.disabled = bool(
            int(os.environ.get("GLM_TOOL_GUARD_DISABLE", "0") or 0)
        )
        self.budget = int(
            os.environ.get("GLM_TOOL_GUARD_MIN_TOKENS_AFTER_THINK", "16") or 16
        )
        # batch_idx → {output_tok_ids: list, think_end_idx: int|None, opened: bool,
        #              announced_armed: bool, masked_steps: int}
        self._state: dict[int, dict] = {}
        self._stop_ids = list(GLM_STOP_IDS)
        self._neg_inf = torch.tensor(
            -float("inf"), dtype=torch.float32, device=device
        )
        self._logits_slice: tuple[torch.Tensor, torch.Tensor] | None = None
        self._stats = {
            "requests_seen": 0,
            "rescues": 0,           # </think> seen, masked, then <tool_call> opened within budget
            "passthrough_text": 0,  # </think> seen, budget exhausted, plain-text answer let through
            "apply_calls": 0,
            "rows_masked_total": 0,
        }
        logger.warning(  # warning so it shows up in default-level log
            "[glm51_nsa_mtp] GUARD INSTANTIATED  device=%s budget=%d disabled=%s  "
            "stops=%s think_end=%d tool_call_open=%d",
            device, self.budget, self.disabled, GLM_STOP_IDS,
            GLM_THINK_END_ID, GLM_TOOL_CALL_OPEN_ID,
        )

    @classmethod
    def validate_params(cls, sampling_params) -> None:
        return None

    def is_argmax_invariant(self) -> bool:
        # We censor tokens, so argmax outcomes can change.
        return False

    @staticmethod
    def _add_request(params, prompt_tok_ids, output_tok_ids):
        return {
            "output_tok_ids": output_tok_ids,  # live reference
            "think_end_idx": None,
            "opened": False,
            "announced_armed": False,
            "masked_steps": 0,
        }

    def update_state(self, batch_update) -> None:
        if self.disabled:
            self._logits_slice = None
            return

        # Apply add/remove/move via the v1 helper if available
        try:
            from vllm.v1.sample.logits_processor.state import (
                process_dict_updates,
            )
        except Exception:
            process_dict_updates = None  # type: ignore

        if batch_update is not None:
            n_added = len(batch_update.added or [])
            n_removed = len(batch_update.removed or [])
            if n_added or n_removed:
                self._stats["requests_seen"] += n_added
                logger.debug(
                    "[glm51_nsa_mtp] guard batch_update +%d -%d  active=%d",
                    n_added, n_removed, len(self._state),
                )
            if process_dict_updates is not None:
                process_dict_updates(self._state, batch_update, self._add_request)
            else:
                for tup in (batch_update.added or []):
                    idx, params, prompt_ids, out_ids = tup
                    self._state[idx] = self._add_request(
                        params, prompt_ids, out_ids
                    )
                for idx in (batch_update.removed or []):
                    self._state.pop(idx, None)

        # Refresh per-request status using live output_tok_ids refs.
        rows: list[int] = []
        cols: list[int] = []
        budget = self.budget
        for idx, st in self._state.items():
            if st["opened"]:
                continue
            out = st["output_tok_ids"]
            # Once <tool_call> appears, disengage permanently.
            if GLM_TOOL_CALL_OPEN_ID in out:
                st["opened"] = True
                if st["masked_steps"] > 0:
                    self._stats["rescues"] += 1
                    logger.info(
                        "[glm51_nsa_mtp] guard RESCUE req=%d  masked_steps=%d  "
                        "→ <tool_call> opened  totals=%s",
                        idx, st["masked_steps"], self._stats,
                    )
                continue
            if st["think_end_idx"] is None:
                if GLM_THINK_END_ID in out:
                    try:
                        st["think_end_idx"] = out.index(GLM_THINK_END_ID)
                    except ValueError:
                        st["think_end_idx"] = len(out)
            if st["think_end_idx"] is None:
                # Pre-reasoning — don't mask, model may legitimately stop
                continue
            steps_since_think = len(out) - st["think_end_idx"]
            if 0 <= steps_since_think < budget:
                if not st["announced_armed"]:
                    st["announced_armed"] = True
                    logger.debug(
                        "[glm51_nsa_mtp] guard ARMED req=%d  "
                        "</think> at idx=%d  steps_since=%d  budget=%d",
                        idx, st["think_end_idx"], steps_since_think, budget,
                    )
                st["masked_steps"] += 1
                for sid in self._stop_ids:
                    rows.append(idx)
                    cols.append(sid)
            elif st["announced_armed"]:
                # Budget exhausted, model produced text → let stops through
                self._stats["passthrough_text"] += 1
                st["announced_armed"] = False  # silence further messages
                logger.debug(
                    "[glm51_nsa_mtp] guard PASSTHROUGH req=%d  "
                    "budget exhausted (no <tool_call>) — plain-text answer",
                    idx,
                )

        if rows:
            self._logits_slice = (
                torch.tensor(rows, device=self.device, dtype=torch.int64),
                torch.tensor(cols, device=self.device, dtype=torch.int64),
            )
        else:
            self._logits_slice = None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.disabled or self._logits_slice is None:
            return logits
        try:
            n_rows = self._logits_slice[0].shape[0]
            logits.index_put_(self._logits_slice, self._neg_inf)
            self._stats["apply_calls"] += 1
            self._stats["rows_masked_total"] += int(n_rows)
        except Exception as e:
            logger.warning("[glm51_nsa_mtp] guard apply() failed: %s", e)
        return logits


# Backwards-compatibility alias for clients importing the legacy single-request shape
make_glm51_tool_emission_guard = GlmToolEmissionGuard

