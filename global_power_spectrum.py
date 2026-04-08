# Compute temporal power spectrum (PSD) of lead-time series for target and predictions
# Mirrors the structure of global_mean_old.py but computes and plots PSD instead of time series.

import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
from typing import List, Tuple


def _open_store(path: str):
    if path.endswith(".zip"):
        return zarr.storage.ZipStore(path)
    return zarr.storage.LocalStore(path)


def _frame_worker(i: int, zarr_files: List[str], var_idx: int) -> Tuple[int, float, List[float]]:
    """
    Worker executed in subprocess: opens stores, computes spatial means for target (first store)
    and predictions (all stores) at lead index i. Returns (i, target_mean, [pred_means...]).
    """
    try:
        stores = [_open_store(p) for p in zarr_files]
        dss = [zarr.open(store=st) for st in stores]

        # read target from first dataset
        try:
            tgt = dss[0][f"0/ERA5/{i}/target/data"][:]  # (npoints, nvars)
            target_mean = float(np.nanmean(tgt[:, var_idx]))
        except Exception:
            target_mean = float("nan")

        pred_means = []
        for ds in dss:
            try:
                pr = ds[f"0/ERA5/{i}/prediction/data"][:]
                if pr.ndim > 2:
                    pr = np.mean(pr, axis=-1)
                pred_means.append(float(np.nanmean(pr[:, var_idx])))
            except Exception:
                pred_means.append(float("nan"))

        for st in stores:
            try:
                st.close()
            except Exception:
                pass

        return (i, target_mean, pred_means)
    except Exception:
        return (i, float("nan"), [float("nan")] * len(zarr_files))


def compute_psd_from_timeseries(x: np.ndarray, dt_hours: float = 6.0):
    """
    Compute one-sided power spectrum using rfft. Returns frequencies in cycles/day and PSD.
    Assumptions and notes:
    - x: 1D array of length N (can contain NaNs). NaNs are linearly interpolated where possible.
    - dt_hours: sampling interval in hours between successive samples (default 6 h as in original script).

    Output:
    - freqs_cpd: frequencies in cycles per day (1/day)
    - psd: power spectral density (power per frequency bin)
    """
    # handle NaNs by simple interpolation; if all NaN return empty arrays
    if np.all(np.isnan(x)):
        return np.array([]), np.array([])

    # interpolate small gaps
    n = len(x)
    x = x.astype(np.float64)
    nan_idx = np.isnan(x)
    if np.any(nan_idx):
        good = ~nan_idx
        if good.sum() < 2:
            # not enough points to interpolate
            x[nan_idx] = 0.0
        else:
            xp = np.flatnonzero(good)
            fp = x[good]
            xi = np.flatnonzero(nan_idx)
            x[nan_idx] = np.interp(xi, xp, fp)

    # subtract mean (detrend simple)
    x = x - np.mean(x)

    # compute rfft
    X = np.fft.rfft(x)
    psd = (np.abs(X) ** 2) / n

    freqs_hz = np.fft.rfftfreq(n, d=dt_hours)  # cycles per hour
    freqs_cpd = freqs_hz * 24.0  # cycles per day

    return freqs_cpd, psd


if __name__ == "__main__":
    # config matching global_mean_old.py default choices
    model_ids = ["h7xsg5m4"]  # adjust as needed
    zarr_files = [f"/e/scratch/weatherai/shared_work/results/{model_id}/validation_chkpt00000_rank0000.zip" for model_id in model_ids]

    var = "2t"
    var_index = 3

    # frames to process (same as original)
    start_lead = 1
    end_lead = 120
    leads = list(range(start_lead, end_lead))

    requested_workers = 220
    max_avail = max(1, multiprocessing.cpu_count())
    workers = min(requested_workers, max_avail)
    print(f"Using workers={workers} (requested {requested_workers}, available {max_avail})")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_frame_worker, i, zarr_files, var_index): i for i in leads}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)

    results.sort(key=lambda x: x[0])
    leads_sorted = [r[0] for r in results]
    target_means = np.array([r[1] for r in results], dtype=np.float64)
    pred_means_all = np.array([r[2] for r in results], dtype=np.float64)  # shape (n_leads, n_models)

    # sampling interval in hours between successive lead indices
    # original script used lead_time = leads_sorted * 6, so dt=6 hours
    dt_hours = 6.0

    # compute PSD for target and each prediction
    freqs, target_psd = compute_psd_from_timeseries(target_means, dt_hours=dt_hours)

    pred_psds = []
    n_preds = pred_means_all.shape[1]
    for j in range(n_preds):
        freqs_j, psd_j = compute_psd_from_timeseries(pred_means_all[:, j], dt_hours=dt_hours)
        pred_psds.append(psd_j)

    pred_psds = np.array(pred_psds) if len(pred_psds) > 0 else np.empty((0, len(freqs)))

    # plotting: PSD vs cycles per day
    plt.figure(figsize=(10, 6))
    if freqs.size > 0 and target_psd.size > 0:
        plt.loglog(freqs, target_psd, color="black", label="target", linewidth=1.5)

        cmap = plt.get_cmap("viridis")
        for j in range(n_preds):
            col = cmap(j / max(1, n_preds - 1))
            if pred_psds.shape[1] == freqs.size:
                plt.loglog(freqs, pred_psds[j, :], color=col, linestyle="-", label=f"pred_v{j+1}", linewidth=1.0)

    plt.xlabel("frequency (cycles per day)")
    plt.ylabel("power")
    plt.title(f"Temporal Power Spectrum of {var} (spatial mean) for models: {','.join(model_ids)}")
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()

    out_fname = f"psd_pred_vs_target_{'_'.join(model_ids)}_{var}_season.png"
    plt.savefig(out_fname, dpi=150)
    print(f"Saved {out_fname}")

    # save numeric PSDs for later use
    np.savez_compressed(f"psd_data_{'_'.join(model_ids)}_{var}.npz", freqs=freqs, target_psd=target_psd, pred_psds=pred_psds)
    print("Saved numeric PSDs to .npz")
