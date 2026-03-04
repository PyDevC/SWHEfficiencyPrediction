import torch
from train import train_model
from eval import evaluate_model
from SWH.utils import DATA_PATH
import os

def main():
    datapth = os.path.join(DATA_PATH, "solar_water_heater_efficiency.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, model_path = train_model(datapth, epochs=30, device=device)
    
    evaluate_model(datapth, model_path, device=device)

if __name__ == "__main__":
    main()
