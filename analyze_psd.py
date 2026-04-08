"""
Analyze PSD produced by global_power_spectrum.py

Usage (from repo root):
    python analyze_psd.py /absolute/path/to/psd_data_h7xsg5m4_2t.npz

Outputs (written to ./analysis_outputs/):
 - psd_compare_<id>_<var>.png       : log-log PSD plot with slope fits
 - psd_ratio_<id>_<var>.png         : ratio (pred/target) vs frequency
 - psd_summary_<id>_<var>.csv       : CSV with numeric metrics per series
 - psd_analysis_<id>_<var>.npz      : compressed NPZ with extracted metrics

If no argument is given, the script will try to find a file matching psd_data_*.npz in the current directory.
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def find_npz(path_arg: str = None) -> str:
    if path_arg:
        if os.path.isfile(path_arg):
            return path_arg
        raise FileNotFoundError(path_arg)
    # try to find matching file in cwd
    matches = glob.glob("./psd_data_*.npz")
    if not matches:
        matches = glob.glob("psd_data_*.npz")
    if not matches:
        raise FileNotFoundError("No psd_data_*.npz found in current directory. Provide a path as argument.")
    # prefer the first match
    return matches[0]


def band_integral(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    # integrate PSD between fmin and fmax (simple trapezoidal rule)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def spectral_slope(freqs: np.ndarray, psd: np.ndarray, fit_min: float, fit_max: float) -> Tuple[float, float]:
    # fit linear slope to log-log PSD in given freq range
    mask = (freqs >= fit_min) & (freqs <= fit_max) & (psd > 0)
    if mask.sum() < 2:
        return float('nan'), float('nan')
    xf = np.log10(freqs[mask])
    yf = np.log10(psd[mask])
    A = np.vstack([xf, np.ones_like(xf)]).T
    m, c = np.linalg.lstsq(A, yf, rcond=None)[0]
    return float(m), float(c)


def analyze(psd_file: str, out_dir: str = "analysis_outputs") -> Dict:
    arr = np.load(psd_file)
    # expected keys: freqs, target_psd, pred_psds
    freqs = arr.get('freqs')
    target_psd = arr.get('target_psd')
    pred_psds = arr.get('pred_psds')

    if freqs is None:
        raise KeyError('freqs not found in npz')

    if target_psd is None:
        raise KeyError('target_psd not found in npz')

    if pred_psds is None:
        # allow shape (n_preds, n_freqs) or (n_freqs,) single pred
        pred_psds = np.empty((0, freqs.size))

    # ensure shapes
    freqs = np.array(freqs)
    target_psd = np.array(target_psd)
    pred_psds = np.array(pred_psds)
    if pred_psds.ndim == 1:
        pred_psds = pred_psds[np.newaxis, :]

    n_preds = pred_psds.shape[0]

    # create out dir
    os.makedirs(out_dir, exist_ok=True)

    # metrics per series
    metrics = []

    # define bands in cycles/day
    bands = [
        (0.0, 1.0/7.0, '>7d'),
        (1.0/7.0, 1.0, '1d-7d'),
        (1.0, np.max(freqs), '<1d')
    ]

    # slope fit range
    fit_min, fit_max = 0.05, 1.0  # cpd

    # analyze target
    def analyze_one(name: str, psd: np.ndarray):
        dom_idx = np.nanargmax(psd)
        dom_freq = float(freqs[dom_idx])
        total_power = float(np.trapz(psd, freqs))
        band_powers = {label: band_integral(freqs, psd, fmin, fmax) for (fmin, fmax, label) in bands}
        slope, intercept = spectral_slope(freqs, psd, fit_min, fit_max)
        return {
            'name': name,
            'dominant_freq_cpd': dom_freq,
            'total_power': total_power,
            'slope_loglog': slope,
            'slope_intercept': intercept,
            **{f'band_power_{label}': v for label, v in zip([b[2] for b in bands], band_powers.values())}
        }

    metrics.append(analyze_one('target', target_psd))

    for i in range(n_preds):
        metrics.append(analyze_one(f'pred_{i+1}', pred_psds[i]))

    # compute ratio pred/target curves
    ratios = []
    for i in range(n_preds):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = pred_psds[i] / target_psd
            ratio = np.where(np.isfinite(ratio), ratio, np.nan)
        ratios.append(ratio)

    # save CSV summary
    import csv
    basename = os.path.splitext(os.path.basename(psd_file))[0]
    csv_file = os.path.join(out_dir, f'psd_summary_{basename}.csv')
    keys = ['name','dominant_freq_cpd','total_power','slope_loglog','slope_intercept'] + [f'band_power_{b[2]}' for b in bands]
    with open(csv_file, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in metrics:
            # ensure all keys present
            outrow = {k: row.get(k, '') for k in keys}
            writer.writerow(outrow)

    # plotting
    plt.figure(figsize=(10,6))
    plt.loglog(freqs, target_psd, label='target', color='black', linewidth=1.8)
    cmap = plt.get_cmap('viridis')
    for i in range(n_preds):
        plt.loglog(freqs, pred_psds[i], label=f'pred_{i+1}', color=cmap(i/max(1,n_preds-1)), linewidth=1.2)

    # overlay slope fit line for target
    slope, intercept = spectral_slope(freqs, target_psd, fit_min, fit_max)
    if np.isfinite(slope):
        xf_mask = (freqs >= fit_min) & (freqs <= fit_max)
        xf = freqs[xf_mask]
        yf_fit = 10**(intercept) * (xf ** slope)
        plt.loglog(xf, yf_fit, color='k', linestyle='--', label=f'target fit slope={slope:.2f}')

    plt.xlabel('frequency (cycles per day)')
    plt.ylabel('power')
    plt.title(f'PSD comparison: {basename}')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f'psd_compare_{basename}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

    # ratio plot
    plt.figure(figsize=(10,4))
    for i, ratio in enumerate(ratios):
        plt.semilogx(freqs, ratio, label=f'pred_{i+1}/target', color=cmap(i/max(1,n_preds-1)))
    plt.axhline(1.0, color='k', lw=0.7, ls='--')
    plt.xlabel('frequency (cycles per day)')
    plt.ylabel('power ratio')
    plt.title(f'PSD ratio: {basename}')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    out_ratio = os.path.join(out_dir, f'psd_ratio_{basename}.png')
    plt.savefig(out_ratio, dpi=150)
    plt.close()

    # save numeric results
    out_npz = os.path.join(out_dir, f'psd_analysis_{basename}.npz')
    np.savez_compressed(out_npz, freqs=freqs, target_psd=target_psd, pred_psds=pred_psds, metrics=metrics)

    summary = {
        'basename': basename,
        'csv': csv_file,
        'psd_plot': out_png,
        'ratio_plot': out_ratio,
        'npz': out_npz,
        'metrics': metrics
    }
    return summary


if __name__ == '__main__':
    try:
        path = sys.argv[1] if len(sys.argv) > 1 else None
        npzfile = find_npz(path)
        print('Analyzing', npzfile)
        res = analyze(npzfile)
        print('Wrote:', res['csv'])
        print('Plots:', res['psd_plot'], res['ratio_plot'])
        print('Numeric results:', res['npz'])
    except Exception as e:
        print('Error:', e)
        sys.exit(2)
