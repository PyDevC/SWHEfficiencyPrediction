import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score

import os

DATA_PATH = os.path.abspath(os.path.realpath(
        os.path.join(os.path.dirname(__file__), "data")
))

def calculate_metrics(y_true, y_pred):
    """Calculate all regression metrics."""
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    
    return {
        "R2": r2,
        "explained variance score": evs,
        "corr": corr,
        "MAE": mae,
        "MAPE": mape
    }

def plot_performance(y_true, y_pred, metrics):
    """Generates the performance bar charts and scatter plots."""
    labels = ['R² Score', 'Explained Variance', 'Pearson (r)']
    scores = [metrics['r2'], metrics['evs'], metrics['corr']]

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    bars = plt.bar(labels, scores, color=['#4CAF50', '#2196F3', '#FF9800'])
    plt.ylim(0, 1.2)
    plt.title('Model Reliability Scores')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.3f}', ha='center')

    plt.subplot(1, 2, 2)
    plt.bar(['MAPE (%)'], [metrics['mape']], color='#f44336', width=0.4)
    plt.title('Average Percentage Error (MAPE)')
    plt.text(0, metrics['mape'] + 1, f"{metrics['mape']:.2f}%", ha='center')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, color='teal')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Efficiency %')
    plt.ylabel('Predicted Efficiency %')
    plt.title('Actual vs. Predicted')
    plt.grid(True, alpha=0.3)
    plt.show()

def save_model(model, filename, save_model_dict=False):
    """Saves the model in the desiredpath"""
    if save_model_dict:
        torch.save(model.state_dict(), filename)
    else:
        torch.save(model, filename + "-full")


def load_model(path, weights_only=False):
    if "-full" in path:
        return torch.load(path, weights_only)
    return torch.load_state_dict(path)
