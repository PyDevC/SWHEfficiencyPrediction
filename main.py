import torch

from train import train_model
from eval import evaluate_model
from SWH.utils import DATA_PATH

import os


datapth = os.path.join(DATA_PATH, "SWHDataset" ,"solar_water_heater_efficiency.csv")
device = "cuda" # Choose your own device
model, model_path = train_model(datapth, epochs=30, device=device)
evaluate_model(datapth, model_path, device=device, model=model)
