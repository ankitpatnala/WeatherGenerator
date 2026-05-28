#!/bin/bash -x
#SBATCH --account=weatherai
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

source .venv/bin/activate

srun uv --offline run inference --from-run-id $1 --start-date=2022-04-01 --options validation_config.samples_per_mini_epoch=1 validation_config.output.num_samples=1 'validation_config.output.streams=[ERA5]' training_config.forecast.num_steps=14 training_config.forecast.forecast_chunk_size=4 validation_config.start_date=2023-01-01T00:00  valiation_config.end_date='2023-21-31' model_path="/e/scratch/weatherai/shared_work/models"
