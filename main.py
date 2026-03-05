import torch

from train import train_model
from eval import evaluate_model, plot_loss_curve, plot_regression_results, display_performance_table
from SWH.utils import DATA_PATH
from SWH.dataset import SolarFWH, get_dataloaders
import pandas as pd

import os

datapth = os.path.join(DATA_PATH, "SWHDataset" ,"Processed_FHW_2017.csv")
df = pd.read_csv(datapth)
device = "cuda" # Choose your own device

def transform_data(X_np, scaler):
    """
    Transforms a numpy array (or tensor) using a pre-fitted scaler.
    Ensures input is 2D for the scaler, then returns a float32 tensor.
    """
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)
    
    transformed = scaler.transform(X_np).astype('float32')
    return torch.tensor(transformed).squeeze(0)


train_loader, test_loader = get_dataloaders(df, batch_size=16, transform_scale=transform_data)
model, model_path, loss_history = train_model(train_loader, epochs=5, device=device)
metrics, targets, preds = evaluate_model(model, test_loader, device=device)
plot_regression_results(targets, preds)
plot_loss_curve(loss_history)
display_performance_table(metrics)
