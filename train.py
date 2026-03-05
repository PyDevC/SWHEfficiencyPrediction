import torch
import torch.nn as nn
import torch.optim as optim
from SWH.model import SolarEfficiencyANNFWH
from SWH.utils import save_model
import os
from tqdm import tqdm

def train_model(train_loader, epochs=100, lr=0.001, device='cpu'):
    os.makedirs("experiments/models", exist_ok=True)
    
    model = SolarEfficiencyANNFWH().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0.0
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(loop):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
            
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")
            
    save_path = "experiments/models/solar_model.pth"
    save_model(model, save_path, save_model_dict=True)
    
    return model, save_path, loss_history
