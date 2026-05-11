import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import glob
import os
import re
import argparse

def get_latents_and_idx(file_path):
    data = torch.load(file_path, map_location='cpu')
    latents = data['latent']
    idx = data['idx']
    return latents, idx

def _idx_to_datetime(idx, base_time=pd.Timestamp("2014-01-01 00:00:00"), time_window=6):
    return ( base_time + pd.Timedelta(hours=idx * time_window),
            base_time + pd.Timedelta(hours=(idx + 1) * time_window)) 

def pick_random_latents(sorted_files, num_samples=1000):
    if len(sorted_files) < num_samples:
        raise ValueError("Not enough latents to sample from.")
    
    random_indices = np.random.choice(len(sorted_files), num_samples, replace=False)
    sampled_latents = []
    sampled_datetime = []
    sampled_indices = []
    for i in random_indices:
        latents, idx = get_latents_and_idx(sorted_files[i])
        sampled_latents.append(latents)
        sampled_datetime.append(_idx_to_datetime(idx.item()))
        sampled_indices.append(idx.item())
    sampled_latents = torch.stack(sampled_latents)

    return sampled_latents, sampled_datetime, sampled_indices

def cluster_latents(latents):
    pass  # Placeholder for clustering implementation

def visualize_idx(
        indices,
        variables, 
        anemoi_folder="/p/scratch/weatherai/shared/weather_generator_data/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
    ):
    import anemoi.datasets as datasets
    import cartopy.crs as ccrs

    ds = datasets.open_dataset(anemoi_folder, start="2014-01-01", end="2022-12-31")
    anemoi_variables = ds.variables
    var_indices = [anemoi_variables.index(var) for var in variables]
    latitudes = ds.latitudes
    longitudes = ds.longitudes

    for idx in indices:
        # data shape: (len(variables), n_nodes)
        data = np.squeeze(ds[idx, var_indices])
        print(f"data shape for idx {idx}: {data.shape}")
        for var_idx, var in enumerate(variables):
            out_dir = f"plots/{var}"
            os.makedirs(out_dir, exist_ok=True)

            fig = plt.figure(figsize=(10, 5))
            ax = plt.axes(projection=ccrs.Robinson())
            sc = ax.scatter(longitudes, latitudes, c=data[var_idx], transform=ccrs.PlateCarree(), cmap='YlOrRd', s=1)
            ax.coastlines()
            plt.title(f"{var} at idx {idx} : {_idx_to_datetime(idx)}")
            plt.colorbar(sc, ax=ax, label=var)
            plt.savefig(f"{out_dir}/idx_{idx}.png")
            plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Extractor")
    parser.add_argument("--folder_path", 
            type=str,
            default="/p/scratch/weatherai/slurm/slurm_weathergen_atmosfo2_copy_dir/WeatherGenerator/latents_2/*.pt", 
            help="Path to the folder containing latent files")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of random latents to sample")
    args = parser.parse_args()

    folder_path = args.folder_path
    num_samples = args.num_samples
    files = sorted(glob.glob(folder_path), key=lambda f: int(re.search(r'\d+', os.path.basename(f)).group()))
    sampled_latents, sampled_datetime, sampled_indices = pick_random_latents(files, num_samples=num_samples)
    print("Sampled Latents Shape:", sampled_latents.shape)
    print("Sampled Datetime Count:", len(sampled_datetime))
    print("Sampled Indices Count:", len(sampled_indices))

    visualize_idx(sampled_indices, variables=['2t', '10u', '10v', 'tp'])
