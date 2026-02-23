#!/usr/bin/env python3

import argparse
import glob
import os
import pathlib
import sys
import tempfile

import numpy as np
import torch


def _load_rank_files(pattern: str, feature_key: str):
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No feature files matched pattern: {pattern}")

    idx_all = []
    feat_all = []
    for path in paths:
        data = np.load(path)
        if "idx" not in data:
            raise KeyError(f"Missing 'idx' in feature file: {path}")
        if feature_key not in data:
            raise KeyError(f"Missing '{feature_key}' in feature file: {path}")
        idx_all.append(np.asarray(data["idx"], dtype=np.int64).reshape(-1))
        feat_all.append(np.asarray(data[feature_key], dtype=np.float32))

    idx = np.concatenate(idx_all, axis=0)
    feat = np.concatenate(feat_all, axis=0)
    return idx, feat


def _deduplicate_by_idx(idx: np.ndarray, feat: np.ndarray, policy: str):
    unique_idx, inverse, counts = np.unique(idx, return_inverse=True, return_counts=True)
    dup_count = int(np.sum(counts > 1))
    if dup_count == 0:
        # Preserve original ordering when there are no duplicates.
        return idx, feat, 0

    if policy == "error":
        raise ValueError(
            f"Found {dup_count} duplicated idx values across feature files. "
            "Use --dedup-policy mean/first to resolve explicitly."
        )

    if policy == "first":
        first_positions = np.full(unique_idx.shape[0], -1, dtype=np.int64)
        for i, uidx in enumerate(inverse):
            if first_positions[uidx] == -1:
                first_positions[uidx] = i
        return unique_idx, feat[first_positions], dup_count

    if policy == "mean":
        d = feat.shape[1]
        feat_sum = np.zeros((unique_idx.shape[0], d), dtype=np.float64)
        np.add.at(feat_sum, inverse, feat.astype(np.float64))
        feat_mean = feat_sum / counts[:, None]
        return unique_idx, feat_mean.astype(np.float32), dup_count

    raise ValueError(f"Unknown dedup policy: {policy}")


def _normalize_features(x: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), eps, None)


def _whiten_features(
    cand_feat: np.ndarray, tgt_feat: np.ndarray, reg: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    """Joint Cholesky whitening of candidate and target features.

    Mirrors the whitening step in TAROT's WFDEstimator.get_wfd():
      1. Concatenate and center (joint mean subtraction).
      2. Compute biased covariance: X^T X / N.
      3. Regularize: add reg * I.
      4. Cholesky-factor and invert: W = inv(L).T.
      5. Apply: X_whitened = X_centered @ W.

    The caller is responsible for L2-normalizing the returned features
    before passing them to DataSelector or the score matrix function.
    """
    n_cand = cand_feat.shape[0]
    all_feat = np.concatenate([cand_feat, tgt_feat], axis=0).astype(np.float64)
    all_feat -= all_feat.mean(axis=0, keepdims=True)
    n = all_feat.shape[0]
    xtx = (all_feat.T @ all_feat) / n
    xtx += np.eye(xtx.shape[0]) * reg
    L = np.linalg.cholesky(xtx)
    W = np.linalg.inv(L).T
    all_feat = (all_feat @ W).astype(np.float32)
    return all_feat[:n_cand], all_feat[n_cand:]


def _subsample_target(
    tgt_idx: np.ndarray, tgt_feat: np.ndarray, max_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or tgt_idx.shape[0] <= max_samples:
        return tgt_idx, tgt_feat
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(tgt_idx.shape[0], size=max_samples, replace=False))
    return tgt_idx[sel], tgt_feat[sel]


def _cosine_score_matrix(
    candidate: np.ndarray,
    target: np.ndarray,
    block_size: int,
    score_memmap_path: str | None,
) -> np.ndarray:
    candidate_norm = _normalize_features(candidate.astype(np.float32, copy=False))
    target_norm = _normalize_features(target.astype(np.float32, copy=False))
    n_cand = candidate_norm.shape[0]
    n_tgt = target_norm.shape[0]

    if score_memmap_path is None:
        score = np.empty((n_cand, n_tgt), dtype=np.float32)
    else:
        score_path = pathlib.Path(score_memmap_path)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score = np.memmap(score_path, dtype=np.float32, mode="w+", shape=(n_cand, n_tgt))

    for i0 in range(0, n_cand, block_size):
        i1 = min(i0 + block_size, n_cand)
        score[i0:i1] = candidate_norm[i0:i1] @ target_norm.T

    return score


def main():
    parser = argparse.ArgumentParser(description="Select TAROT indices from exported WG features.")
    parser.add_argument("--candidate", required=True, help="Glob for candidate npz files.")
    parser.add_argument("--target", required=True, help="Glob for target npz files.")
    parser.add_argument(
        "--feature",
        default="mean_std",
        choices=["mean", "mean_std", "grad_projected"],
        help="Feature key to use from exported npz files.",
    )
    parser.add_argument("--ratio", type=float, default=0.2, help="Selection ratio in (0,1].")
    parser.add_argument("--output", required=True, help="Output .npy path for selected idx.")
    parser.add_argument(
        "--method",
        default="fixed_size",
        choices=["fixed_size", "random", "dsdm", "less", "otm"],
        help="TAROT selection method.",
    )
    parser.add_argument(
        "--dedup-policy",
        default="mean",
        choices=["mean", "first", "error"],
        help="How to handle duplicated idx across matched feature files.",
    )
    parser.add_argument(
        "--target-max-samples",
        type=int,
        default=0,
        help="If >0, subsample target features to this many samples before score computation.",
    )
    parser.add_argument(
        "--target-subsample-seed",
        type=int,
        default=42,
        help="Random seed for target subsampling.",
    )
    parser.add_argument(
        "--score-block-size",
        type=int,
        default=4096,
        help="Block size along candidate axis for cosine score matrix computation.",
    )
    parser.add_argument(
        "--score-memmap-path",
        default=None,
        help=(
            "Optional path to store score matrix as memmap (reduces RAM pressure, uses disk). "
            "If omitted, score is kept in RAM."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {args.ratio}")
    if args.score_block_size <= 0:
        raise ValueError("--score-block-size must be > 0")

    tarot_root = pathlib.Path(__file__).resolve().parent.parent / "TAROT"
    sys.path.insert(0, str(tarot_root))
    from tarot.data_selector import DataSelector

    feature_key = f"feat_{args.feature}"
    cand_idx, cand_feat = _load_rank_files(args.candidate, feature_key)
    tgt_idx, tgt_feat = _load_rank_files(args.target, feature_key)
    cand_idx, cand_feat, cand_dups = _deduplicate_by_idx(cand_idx, cand_feat, args.dedup_policy)
    tgt_idx, tgt_feat, tgt_dups = _deduplicate_by_idx(tgt_idx, tgt_feat, args.dedup_policy)
    if cand_dups > 0 or tgt_dups > 0:
        print(
            f"Deduplicated repeated idx values with policy={args.dedup_policy}: "
            f"candidate_duplicates={cand_dups}, target_duplicates={tgt_dups}"
        )
    tgt_idx, tgt_feat = _subsample_target(
        tgt_idx, tgt_feat, args.target_max_samples, args.target_subsample_seed
    )
    if args.target_max_samples > 0:
        print(f"Target subsampling active: using {tgt_idx.shape[0]} target samples.")

    # Joint Cholesky whitening (mirrors WFDEstimator.get_wfd).
    # After whitening, apply per-row L2 normalization so that the score matrix
    # equals the cosine similarity in the whitened space, and the OT distance
    # in DataSelector's cosine_L2 is the correct sqrt(2 - 2·cos) metric.
    print(f"Whitening features: candidate={cand_feat.shape}, target={tgt_feat.shape}")
    cand_feat, tgt_feat = _whiten_features(cand_feat, tgt_feat)
    cand_feat_norm = _normalize_features(cand_feat)
    tgt_feat_norm = _normalize_features(tgt_feat)
    print("Whitening done.")

    score_memmap_path = args.score_memmap_path
    tmp_memmap = None
    if score_memmap_path is None and cand_feat.shape[0] * tgt_feat.shape[0] > 40_000_000:
        tmp_memmap = tempfile.NamedTemporaryFile(
            prefix="tarot_score_", suffix=".mmap", delete=False
        )
        tmp_memmap.close()
        score_memmap_path = tmp_memmap.name
        print(
            f"Large score matrix detected ({cand_feat.shape[0]} x {tgt_feat.shape[0]}). "
            f"Using temporary memmap file: {score_memmap_path}"
        )

    score = None
    try:
        # cand_feat_norm / tgt_feat_norm are whitened + L2-normalized,
        # matching the features returned by WFDEstimator.get_wfd().
        score = _cosine_score_matrix(
            cand_feat_norm,
            tgt_feat_norm,
            block_size=args.score_block_size,
            score_memmap_path=score_memmap_path,
        )
        cfg = {
            "device": "cpu",
            "selection_method": args.method,
            "selection_ratio": args.ratio,
            "k_fold_splits": 10,
            "merge_target_data": False,
            "data_weighting": False,
        }
        selector = DataSelector(cfg)
        selected_pos, _ = selector.select_data(
            score, torch.from_numpy(cand_feat_norm), torch.from_numpy(tgt_feat_norm)
        )
        selected_idx = np.unique(cand_idx[np.asarray(selected_pos, dtype=np.int64)])
    finally:
        if tmp_memmap is not None:
            if score is not None:
                del score
            if score_memmap_path is not None and os.path.exists(score_memmap_path):
                os.remove(score_memmap_path)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, selected_idx)
    print(
        f"Saved {len(selected_idx)} selected indices to {out_path}. "
        f"candidate={len(cand_idx)} target={len(tgt_idx)} feature={args.feature}"
    )


if __name__ == "__main__":
    main()
