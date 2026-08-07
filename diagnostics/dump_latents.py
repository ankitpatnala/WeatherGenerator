#!/usr/bin/env python
"""
Dump the FE latent (z_pre_norm, post-FE) per forecast step to disk, by wrapping the
standard inference entry point. Latents are written in CHUNKS (one .npy holding a stack of
several steps) to avoid an inode blow-up on long (decadal) rollouts.

  Set A (free-running): one long rollout from a single IC. latent_A[k] = FE^k(encode(real(t0))).
  Set B (truth-anchored): many independent 1-step forecasts. latent_B[k] = FE(encode(real(t_{k-1}))).

Output layout ($LATENT_DUMP_DIR, typically results/<run_id>/latents):
  z_{start:06d}.npy   float16, shape (n_steps_in_chunk, num_cells, dim); start = global step of
                      the first entry. Chunks are flushed every LATENT_FLUSH_EVERY steps.
  meta_<jobid>.json

Chunk flushing is aligned to the flush size, which should equal the rollout chunk_size so a
flushed chunk file matches a checkpointed rollout chunk. On a mid-chunk kill the buffered
(unflushed) steps are lost and the resume simply recomputes that whole chunk -> no gaps, no
half-written steps.

Indexing modes (LATENT_INDEX_MODE):
  rollout : global step = output.step_offset + fstep (resumable long rollout, Set A)
  counter : global step = LATENT_START_INDEX + local_counter (independent samples, Set B)

Usage (same options as the inference CLI):
  LATENT_DUMP_DIR=results/<run_id>/latents python dump_latents.py --from-run-id oq9o0t86 ...
"""

import json
import os
import sys

import numpy as np
import torch

DUMP_DIR = os.environ["LATENT_DUMP_DIR"]
os.makedirs(DUMP_DIR, exist_ok=True)

INDEX_MODE = os.environ.get("LATENT_INDEX_MODE", "counter")
START_INDEX = int(os.environ.get("LATENT_START_INDEX", "0"))
FLUSH_EVERY = int(os.environ.get("LATENT_FLUSH_EVERY", "50"))
# which latent to dump per forecast step:
#   post : z_pre_norm, the FE output (post-FE) -- the default
#   pre  : the FE *input* (pre-FE / assimilation-space latent). For the encoder stream this is
#          encode(real(t)); for a free-running rollout it is the carried latent entering the FE.
CAPTURE = os.environ.get("LATENT_CAPTURE", "post")

import weathergen.model.model as _model_mod  # noqa: E402

_orig_add_latent = _model_mod.ModelOutput.add_latent_prediction
_STATE = {"count": 0, "buf": [], "buf_start": None, "chunks": 0, "steps": 0}
_STASH = {"pre": None}

if CAPTURE == "pre":
    # stash the FE input on every forecast step; it is saved (with the correct global-step
    # index) when add_latent_prediction fires for that same step, just after the FE.
    from weathergen.model.engines import ForecastingEngine  # noqa: E402

    _orig_fe_forward = ForecastingEngine.forward

    def _fe_forward_stash(self, tokens, fstep, coords=None):
        z = tokens.detach()
        _STASH["pre"] = z[0] if z.dim() == 3 else z
        return _orig_fe_forward(self, tokens, fstep, coords)

    ForecastingEngine.forward = _fe_forward_stash


def _flush():
    buf = _STATE["buf"]
    if not buf:
        return
    start = _STATE["buf_start"]
    arr = np.stack(buf, axis=0)  # (n, num_cells, dim), already fp16
    np.save(os.path.join(DUMP_DIR, f"z_{start:06d}.npy"), arr)
    _STATE["chunks"] += 1
    _STATE["steps"] += len(buf)
    _STATE["last_end"] = start + len(buf) - 1
    if "first_start" not in _STATE:
        _STATE["first_start"] = start
        _STATE["shape"] = tuple(arr.shape[1:])
    _STATE["buf"] = []
    _STATE["buf_start"] = None


def _add_latent_dumping(self, fstep, latent_name, pred):
    # add_latent_prediction("latent_state") fires once per forecast step (just after the FE),
    # giving the global step index. We save either the FE output (post) or the stashed FE
    # input (pre), buffer, and chunk it.
    if latent_name == "latent_state" and pred is not None and pred.z_pre_norm is not None:
        if CAPTURE == "pre":
            z = _STASH["pre"]
            if z is None:
                return _orig_add_latent(self, fstep, latent_name, pred)
        else:
            z = pred.z_pre_norm.detach()
            if z.dim() == 3:  # (B, num_tokens, dim); inference uses B=1
                z = z[0]
        if INDEX_MODE == "rollout":
            idx = int(getattr(self, "step_offset", 0)) + int(fstep)
        else:
            idx = START_INDEX + _STATE["count"]
        if not _STATE["buf"]:
            _STATE["buf_start"] = idx
        _STATE["buf"].append(z.to(torch.float16).cpu().numpy())
        _STATE["count"] += 1
        if len(_STATE["buf"]) >= FLUSH_EVERY:
            _flush()
    return _orig_add_latent(self, fstep, latent_name, pred)


_model_mod.ModelOutput.add_latent_prediction = _add_latent_dumping


def _write_meta():
    _flush()  # flush any trailing partial chunk on graceful exit
    meta = {
        "steps_this_job": _STATE["steps"],
        "chunks_this_job": _STATE["chunks"],
        "index_mode": INDEX_MODE,
        "flush_every": FLUSH_EVERY,
        "first_start": _STATE.get("first_start"),
        "last_end": _STATE.get("last_end"),
        "shape_per_step": _STATE.get("shape"),
        "dtype": "float16",
        "argv": sys.argv[1:],
    }
    suffix = os.environ.get("SLURM_JOB_ID", "local")
    with open(os.path.join(DUMP_DIR, f"meta_{suffix}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump_latents] wrote {_STATE['steps']} steps in {_STATE['chunks']} chunks to "
          f"{DUMP_DIR} (idx {_STATE.get('first_start')}..{_STATE.get('last_end')}, mode={INDEX_MODE})")


def main():
    import weathergen.utils.cli as cli
    from weathergen.run_train import main as wg_main

    print(f"[dump_latents] chunked z_pre_norm dump -> {DUMP_DIR} "
          f"(flush_every={FLUSH_EVERY}, mode={INDEX_MODE})")
    try:
        wg_main([cli.Stage.inference] + sys.argv[1:])
    finally:
        _write_meta()


if __name__ == "__main__":
    main()
