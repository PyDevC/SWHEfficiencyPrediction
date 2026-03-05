import torch
import torch.nn as nn
import torch.optim as optim
from SWH.dataset import get_dataloaders
from SWH.model import SolarEfficiencyANN
from SWH.utils import save_model
import os
from tqdm import tqdm

def train_model(data_path, epochs=100, batch_size=32, lr=0.001, device='cpu'):
    os.makedirs("experiments/models", exist_ok=True)
    train_loader, _ = get_dataloaders(data_path, batch_size=batch_size)
    
    model = SolarEfficiencyANN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0.0
        for batch_idx, (features, labels) in enumerate(loop):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")
            
    save_path = "experiments/models/solar_model.pth"
    save_model(model, save_path, save_model_dict=True)
    
    return model, save_path
