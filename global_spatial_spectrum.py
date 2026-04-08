"""
Compute spatial power spectrum (isotropic radial wavenumber PSD) of spatial fields
across lead times, for target and model predictions.

This mirrors the data access pattern used in `global_mean_old.py` / `global_power_spectrum.py`:
- reads arrays from zarr at path: '0/ERA5/{lead}/target/data' and '.../prediction/data'
- input arrays may be 2D (ny, nx) or 1D flattened (npoints, nvars) or (npoints,) -- we try to reshape
  a flattened array to a square grid when possible (npoints is a perfect square). If not, the script
  will attempt to infer shape from a square factorization; otherwise it will raise an error.

Outputs:
- psd_spatial_<modelids>_<var>.npz  : freqs (wavenumber), target_psd_avg, pred_psd_avg
- psd_spatial_compare_<...>.png     : log-log radial wavenumber PSD comparison

Assumptions and notes:
- We compute 2D FFT, take |F|^2, and bin radially in wavenumber k (cycles per grid-length).
- The physical scaling (meters/km) is not known here; results are in cycles per grid spacing.
- If the input arrays have an extra variable dimension (npoints, nvars), the script uses var_index.

Usage example:
    python global_spatial_spectrum.py

Adjust `model_ids`, `zarr_files`, and `var_index` in the __main__ block as needed.
"""

import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
from typing import List, Tuple


def _open_store(path: str):
    if path.endswith('.zip'):
        return zarr.storage.ZipStore(path)
    return zarr.storage.LocalStore(path)


def _read_field_array(arr, var_idx: int = 0):
    """Return a 2D field (ny, nx) from various possible input shapes.
    - If arr.ndim == 2: return as-is
    - If arr.ndim == 1: try to reshape to (n, n) if perfect square; else try to find (ny, nx) by factorization
    - If arr.ndim > 2: average over last axis if it represents ensemble/levels, then recurse.
    """
    a = np.array(arr)
    if a.ndim > 2:
        # average last axis
        a = np.mean(a, axis=-1)

    if a.ndim == 2:
        return a

    if a.ndim == 1:
        n = a.size
        # if shape is (npoints, nvars) and var dimension present, try to select var
        # detect if second dimension exists in original arr shape
        # but here a is 1D; assume flattened single variable
        # try perfect square
        sq = int(np.round(np.sqrt(n)))
        if sq * sq == n:
            return a.reshape((sq, sq))
        # try to find factors close to each other
        # find factor pair (ny,nx) with ny>=nx and ny-nx minimal
        best = None
        for i in range(int(np.sqrt(n)), 0, -1):
            if n % i == 0:
                j = n // i
                best = (i, j)
                break
        if best is not None:
            ny, nx = best
            return a.reshape((ny, nx))
        # cannot reshape
        raise ValueError(f"Cannot infer 2D shape from 1D array of length {n}")

    raise ValueError(f"Unsupported array ndim={a.ndim}")


def _frame_worker_spatial(i: int, zarr_files: List[str], var_idx: int) -> Tuple[int, np.ndarray, List[np.ndarray]]:
    """
    Worker opens zarr files for a specific lead index i and returns (i, target_field, [pred_fields...])
    where fields are 2D numpy arrays (ny, nx).
    """
    try:
        stores = [_open_store(p) for p in zarr_files]
        dss = [zarr.open(store=st) for st in stores]

        # target from first dataset
        target_field = None
        try:
            tgt = dss[0][f"0/ERA5/{i}/target/data"][:]
            # if tgt is (npoints, nvars) and var_idx selects variable:
            if tgt.ndim == 2 and tgt.shape[1] > 1:
                fld = tgt[:, var_idx]
            else:
                fld = tgt
            target_field = _read_field_array(fld, var_idx)
        except Exception:
            target_field = None

        pred_fields = []
        for ds in dss:
            try:
                pr = ds[f"0/ERA5/{i}/prediction/data"][:]
                if pr.ndim > 2 and pr.shape[-1] > 1:
                    # if last axis are variables, pick var_idx along last axis
                    if pr.ndim == 3 and pr.shape[-1] > var_idx:
                        fldp = pr[:, :, var_idx]
                    else:
                        # average if uncertain
                        fldp = np.mean(pr, axis=-1)
                elif pr.ndim == 2 and pr.shape[1] > 1:
                    # shape (npoints, nvars)
                    fldp = pr[:, var_idx]
                else:
                    fldp = pr
                pred_fields.append(_read_field_array(fldp, var_idx))
            except Exception:
                pred_fields.append(None)

        for st in stores:
            try:
                st.close()
            except Exception:
                pass

        return (i, target_field, pred_fields)
    except Exception:
        return (i, None, [None] * len(zarr_files))


def radial_psd_from_2d(field: np.ndarray, dx: float = 1.0, nbins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute isotropic radial power spectral density from 2D field using FFT.
    - dx: grid spacing (arbitrary units); default 1.0 meaning cycles per grid spacing
    Returns (k_centers, psd_radial)
    Units: k in cycles per grid spacing unit (i.e., cycles per grid-length)
    """
    f = np.array(field, dtype=np.float64)
    # detrend simple mean
    f = f - np.nanmean(f)

    ny, nx = f.shape
    # replace NaNs with local mean
    if np.any(np.isnan(f)):
        # fill with mean
        f = np.where(np.isnan(f), np.nanmean(f), f)

    F = np.fft.fft2(f)
    P2 = np.abs(F) ** 2
    # compute frequency grids
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    kx = fx[np.newaxis, :]
    ky = fy[:, np.newaxis]
    K = np.sqrt(kx ** 2 + ky ** 2)

    # flatten
    K_flat = K.ravel()
    P_flat = P2.ravel()

    # max k (Nyquist)
    kmax = np.max(K_flat)
    if nbins is None:
        nbins = int(np.round(min(nx, ny) / 2))
        nbins = max(10, nbins)

    bins = np.linspace(0.0, kmax, nbins + 1)
    bin_idx = np.digitize(K_flat, bins) - 1

    k_centers = 0.5 * (bins[:-1] + bins[1:])
    psd_radial = np.zeros(nbins, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.int64)

    for b in range(nbins):
        mask = bin_idx == b
        if np.any(mask):
            psd_radial[b] = np.mean(P_flat[mask])
            counts[b] = np.count_nonzero(mask)
        else:
            psd_radial[b] = np.nan

    # remove zero-count bins
    valid = counts > 0
    return k_centers[valid], psd_radial[valid]


if __name__ == '__main__':
    # config
    model_ids = ['h7xsg5m4']  # example; replace/add as needed
    zarr_files = [f"/e/scratch/weatherai/shared_work/results/{model_id}/validation_chkpt00000_rank0000.zip" for model_id in model_ids]

    var = '2t'
    var_index = 3

    start_lead = 1
    end_lead = 1440
    leads = list(range(start_lead, end_lead))

    requested_workers = 32
    max_avail = max(1, multiprocessing.cpu_count())
    workers = min(requested_workers, max_avail)
    print(f"Using workers={workers} (requested {requested_workers}, available {max_avail})")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_frame_worker_spatial, i, zarr_files, var_index): i for i in leads}
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception:
                pass

    # sort
    results.sort(key=lambda x: x[0])

    # accumulate radial PSDs per lead and average later
    k_ref = None
    target_psds = []
    pred_psds_all = []  # list of lists per model

    for i, tgt_field, pred_fields in results:
        if tgt_field is None:
            continue
        try:
            k, psd_t = radial_psd_from_2d(tgt_field)
        except Exception:
            continue
        if k_ref is None:
            k_ref = k
        else:
            # if k differs, rebin or skip; for simplicity skip mismatched
            if not np.allclose(k_ref, k):
                # try to interpolate to k_ref
                from numpy import interp
                psd_t = np.interp(k_ref, k, psd_t, left=np.nan, right=np.nan)
        target_psds.append(psd_t)

        # predictions
        if pred_fields is None:
            continue
        if len(pred_psds_all) == 0:
            pred_psds_all = [[None] * 0 for _ in pred_fields]
        # ensure pred_psds_all has entry per pred
        if len(pred_psds_all) < len(pred_fields):
            pred_psds_all = [list() for _ in range(len(pred_fields))]

        for j, pf in enumerate(pred_fields):
            if pf is None:
                pred_psds_all[j].append(np.full_like(k_ref, np.nan))
                continue
            try:
                kpf, psd_p = radial_psd_from_2d(pf)
                if not np.allclose(k_ref, kpf):
                    psd_p = np.interp(k_ref, kpf, psd_p, left=np.nan, right=np.nan)
                pred_psds_all[j].append(psd_p)
            except Exception:
                pred_psds_all[j].append(np.full_like(k_ref, np.nan))

    # average across leads
    if k_ref is None:
        raise RuntimeError('No valid PSDs computed')

    target_psds = np.array(target_psds)  # (n_leads, n_k)
    target_psd_mean = np.nanmean(target_psds, axis=0)

    pred_psd_means = []
    for j in range(len(pred_psds_all)):
        arr = np.array(pred_psds_all[j])
        pred_psd_means.append(np.nanmean(arr, axis=0))
    pred_psd_means = np.array(pred_psd_means)

    # plot
    plt.figure(figsize=(10,6))
    plt.loglog(k_ref, target_psd_mean, color='k', lw=1.8, label='target')
    cmap = plt.get_cmap('viridis')
    for j in range(pred_psd_means.shape[0]):
        plt.loglog(k_ref, pred_psd_means[j], color=cmap(j/max(1,pred_psd_means.shape[0]-1)), lw=1.2, label=f'pred_{j+1}')

    plt.xlabel('wavenumber (cycles per grid length)')
    plt.ylabel('power')
    plt.title(f'Spatial PSD (averaged over leads) for var={var} models={",".join(model_ids)}')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.4)
    plt.tight_layout()

    out_png = f'psd_spatial_compare_{"_".join(model_ids)}_{var}.png'
    plt.savefig(out_png, dpi=150)
    print('Saved', out_png)

    out_npz = f'psd_spatial_{"_".join(model_ids)}_{var}.npz'
    np.savez_compressed(out_npz, k=k_ref, target_psd=target_psd_mean, pred_psds=pred_psd_means)
    print('Saved', out_npz)
