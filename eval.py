import torch
import numpy as np

from SWH.dataset import get_dataloaders
from SWH.model import SolarEfficiencyANN
from SWH.utils import calculate_metrics, plot_performance

def evaluate_model(data_path, model_path, device='cpu', model=None):
    _, test_loader = get_dataloaders(data_path, batch_size=32)
    
    if model is None:
        model = SolarEfficiencyANN(input_head=7).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    
    preds = []
    labels = []
    
    with torch.no_grad():
        for features, label in test_loader:
            features = features.to(device, non_blocking=True)
            out = model(features)
            preds.append(out)
            labels.append(label.to(device))
            
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(labels, dim=0)

    y_pred_c = y_pred.cpu().numpy()
    y_true_c = y_true.cpu().numpy()
    
    metrics = calculate_metrics(y_true_c, y_pred_c)
    print("Evaluation Results")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    plot_performance(y_true_c, y_pred_c, metrics)
