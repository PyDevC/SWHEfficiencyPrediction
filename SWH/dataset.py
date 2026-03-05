import torch
from torch.utils.data import Dataset

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from typing import List
from dataclasses import dataclass, field

@dataclass(frozen=True)
class SolarWHMetaData:
    """
    Schema definition for Solar Water Heater dataset.

    NOTE: Keep the class frozen to avoid data manipulation midway
    """
    features: List[str] = field(default_factory=lambda: [
        'Solar_Radiation_Wm2', 
        'Ambient_Temp_C', 
        'Inlet_Water_Temp_C',
        'Flow_Rate_Lmin', 
        'Collector_Area_m2', 
        'Wind_Speed_ms', 
        'Tilt_Angle_deg'
    ])

    label: str = 'Efficiency_percent'

class SolarWH(Dataset):
    """
    Solar Water Heater Efficiency Prediction Dataset

    Args:
        datapthd: Full path of the .csv file of dataset
        transform: transformation functions to be applied to dataset
        test_size: train_test_split for the dataset, keep this value consistent across both train and test dataset
    """
    def __init__(self, datapth: str, test_size=0.2,  transform=None, train=True):

        self.metadata = SolarWHMetaData()

        df = pd.read_csv(datapth)
        df_clean = df[self.metadata.features + [self.metadata.label]].dropna()

        train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=42)

        subset = pd.DataFrame(train_df) if train else pd.DataFrame(test_df) # Choose one train or test

        features_X = subset[self.metadata.features].values
        label_Y = subset[self.metadata.label].values

        self.features = torch.tensor(features_X, dtype=torch.float32)
        self.label = torch.tensor(label_Y, dtype=torch.float32).reshape(-1, 1)

        if transform is not None:
            # Keeping the variable name same for consistency
            self.features = transform(self.features)
            self.label = transform(self.label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.label[idx]

def transform_scale(X):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X).astype('float32')

def get_dataloaders(datapth, batch_size=32):
    train_dataset = SolarWH(datapth, transform=transform_scale, train=True)
    test_dataset = SolarWH(datapth, transform=transform_scale, train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader
