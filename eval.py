import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device='cpu'):
    """
    Calculates specific metrics mentioned in the report: R2, MAE, MAPE, and Explained Variance.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    preds = np.vstack(all_preds).flatten()
    targets = np.vstack(all_targets).flatten()
    
    metrics = {
        "R2 Score": r2_score(targets, preds),
        "MAE": mean_absolute_error(targets, preds),
        "MAPE": mean_absolute_percentage_error(targets, preds),
        "Explained Variance": explained_variance_score(targets, preds),
        "RMSE": np.sqrt(np.mean((preds - targets)**2))
    }
    return metrics, targets, preds

def plot_loss_curve(loss_history):
    """
    Visualizes the training progression (Loss vs Epoch).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss', color='royalblue', lw=2)
    plt.title('Model Optimization: Loss vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_regression_results(targets, preds):
    """
    Generates regression plots to visualize prediction accuracy.
    """
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=targets, y=preds, alpha=0.5, color='teal')
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Solar Efficiency')
    plt.xlabel('Actual Efficiency / Heat Gain')
    plt.ylabel('Predicted Efficiency / Heat Gain')
    plt.grid(True, alpha=0.2)
    plt.show()

def display_performance_table(metrics):
    """
    Formats the final metric table as requested in section 11.
    """
    df_metrics = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    print("\n--- Final Performance Analysis ---")
    print(df_metrics.to_string(index=False))
