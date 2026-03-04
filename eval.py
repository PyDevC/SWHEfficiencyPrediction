import torch
from SWH.dataset import get_dataloaders
from SWH.model import SolarEfficiencyANN
from SWH.utils import calculate_metrics, plot_performance
import numpy as np

def evaluate_model(data_path, model_path, device='cpu'):
    _, test_loader = get_dataloaders(data_path, batch_size=32)
    
    model = SolarEfficiencyANN(input_head=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    
    metrics = calculate_metrics(y_true, y_pred)
    print("Evaluation Results")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    plot_performance(y_true, y_pred, metrics)
