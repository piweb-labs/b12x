"""sitecustomize hook — monkey-patches vLLM's `build_logitsprocs` to allow
custom logits processors to coexist with speculative decoding (MTP).

WHY this exists
---------------
vLLM v1 rejects user `--logits-processors` whenever `speculative_config` is
set (see `vllm/v1/sample/logits_processor/__init__.py:200-203`). The bundled
`GlmToolEmissionGuard` plugin needs to run alongside MTP to fix the
GLM-5.1 NSA+MTP `no_xml_emitted` bug, so we widen the check here.

HOW the patch works
-------------------
We wrap `build_logitsprocs`:
- pooling models: leave as-is (still rejected)
- spec-decode + no custom procs: leave as-is
- spec-decode + custom procs: instead of raising, return `[MinTokens, *custom]`
  (matching the no-spec-decode branch behaviour minus the other builtins)

LOADING
-------
Python's `site` machinery auto-imports the first `sitecustomize` it finds on
`sys.path`. The vLLM launcher prepends `/home/eddy/桌面/vllm/plugins` to
`PYTHONPATH`, so this file wins in every interpreter (API server + EngineCore
+ each WorkerProc fork).

If the patch ever causes trouble, set env `GLM_DISABLE_LOGITSPROC_PATCH=1`
before launching to skip it.
"""
from __future__ import annotations

import logging
import os
import sys

_log = logging.getLogger("glm51_nsa_mtp.sitecustomize")

_SHOULD_SKIP = bool(int(os.environ.get("GLM_DISABLE_LOGITSPROC_PATCH", "0") or 0))

if _SHOULD_SKIP:
    _log.warning("[glm51 patch] disabled via GLM_DISABLE_LOGITSPROC_PATCH=1")
else:
    try:
        # Import lazily — many Python processes (e.g. quick `python -c` calls)
        # don't have vllm available; we must not crash them.
        try:
            import vllm.v1.sample.logits_processor as _lp_mod
            from vllm.v1.sample.logits_processor import (  # type: ignore
                LogitsProcessors,
                _load_custom_logitsprocs,
            )
            from vllm.v1.sample.logits_processor.builtin import (  # type: ignore
                MinTokensLogitsProcessor,
            )
        except Exception:
            # Not a vllm process — nothing to do
            _lp_mod = None  # type: ignore

        if _lp_mod is not None and not getattr(_lp_mod, "_glm51_patched", False):
            _orig_build = _lp_mod.build_logitsprocs

            def _patched_build(
                vllm_config,
                device,
                is_pin_memory,
                is_pooling_model,
                custom_logitsprocs=(),
            ):
                # Pooling: never allow customs
                if is_pooling_model:
                    return _orig_build(
                        vllm_config, device, is_pin_memory,
                        is_pooling_model, custom_logitsprocs,
                    )
                # Spec-decode + no customs → original behaviour (returns [MinTokens])
                if vllm_config.speculative_config and not custom_logitsprocs:
                    return _orig_build(
                        vllm_config, device, is_pin_memory,
                        is_pooling_model, custom_logitsprocs,
                    )
                # Spec-decode + customs → BYPASS the upstream raise:
                # load custom procs alongside MinTokens and return.
                if vllm_config.speculative_config and custom_logitsprocs:
                    print(
                        "[glm51 patch] build_logitsprocs: bypassing spec-decode "
                        "guard — loading custom logits processors alongside "
                        "MinTokens (procs=%r)" % (list(custom_logitsprocs),),
                        file=sys.stderr, flush=True,
                    )
                    # PRE-REGISTER any classes-by-name as virtual subclasses of
                    # LogitsProcessor. This handles the worker-side identity
                    # mismatch where the class object was pickled from API
                    # server (its base LogitsProcessor has a different id than
                    # the worker's LogitsProcessor). ABC.register() is
                    # identity-independent, so issubclass downstream will pass.
                    from vllm.v1.sample.logits_processor.interface import (
                        LogitsProcessor as _LP_here,
                    )
                    for spec in custom_logitsprocs:
                        if not isinstance(spec, str):
                            cls = spec
                        else:
                            mod_path, _, qual = spec.partition(":")
                            if not qual:  # accept module.Class too
                                mod_path, _, qual = spec.rpartition(".")
                            try:
                                import importlib as _imp
                                mod = _imp.import_module(mod_path)
                                cls = getattr(mod, qual)
                            except Exception as _re:
                                print(
                                    f"[glm51 patch] pre-register: import "
                                    f"failed for {spec!r}: {_re}",
                                    file=sys.stderr, flush=True,
                                )
                                continue
                        if isinstance(cls, type) and not issubclass(cls, _LP_here):
                            _LP_here.register(cls)
                            print(
                                f"[glm51 patch] registered {cls.__name__} as "
                                f"virtual subclass of LogitsProcessor "
                                f"(LP_id={id(_LP_here)} cls_id={id(cls)})",
                                file=sys.stderr, flush=True,
                            )
                    custom_classes = _load_custom_logitsprocs(custom_logitsprocs)
                    return LogitsProcessors(
                        [MinTokensLogitsProcessor(vllm_config, device, is_pin_memory)] +
                        [ctor(vllm_config, device, is_pin_memory) for ctor in custom_classes]
                    )
                # No spec-decode → original (full builtin set + customs)
                return _orig_build(
                    vllm_config, device, is_pin_memory,
                    is_pooling_model, custom_logitsprocs,
                )

            _lp_mod.build_logitsprocs = _patched_build
            _lp_mod._glm51_patched = True
            # Also patch any module that already imported the name.
            try:
                import vllm.v1.worker.gpu_model_runner as _gmr
                if getattr(_gmr, "build_logitsprocs", None) is _orig_build:
                    _gmr.build_logitsprocs = _patched_build
            except Exception:
                pass
            print(
                "[glm51 patch] sitecustomize installed build_logitsprocs wrapper "
                "(spec-decode + custom procs now allowed)  pid=%d" % os.getpid(),
                file=sys.stderr, flush=True,
            )
    except Exception as e:  # pragma: no cover
        # Never break user's Python startup
        print(f"[glm51 patch] sitecustomize FAILED: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
