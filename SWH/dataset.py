import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
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


@dataclass(frozen=True)
class SolarFWHMetaData:
    features: List[str] = field(default_factory=lambda: [
        'rd_gti', 'te_amb', 'te_in', 'vf', 've_wind', 'rh_amb'
    ])
    
    physics_calc_cols: List[str] = field(default_factory=lambda: [
        'mf__calc', 'cp_out__calc', 'te_out', 'te_in'
    ])

    label_name: str = 'efficiency'


class SolarFWH(Dataset):
    def __init__(self, data, scalar=None, transform=None):
        """
        Args:
            data: Can be a file path (str) or a pre-loaded pandas DataFrame.
            transform: A function/transform to apply to the tensors.
        """
        self.metadata = SolarFWHMetaData()
        
        if isinstance(data, str):
            df = pd.read_csv(data, delimiter=";")
        else:
            df = data.copy()

        if self.metadata.label_name not in df.columns:
            df[self.metadata.label_name] = (
                df['mf__calc'] * df['cp_out__calc'] * (df['te_out'] - df['te_in'])
            )

        df = df[df['rd_gti'] > 20].reset_index(drop=True)

        self.X = torch.tensor(df[self.metadata.features].values, dtype=torch.float32)
        self.y = torch.tensor(df[self.metadata.label_name].values.reshape(-1, 1), dtype=torch.float32)

        self.transform = transform
        self.scalar = scalar

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features, label = self.X[idx], self.y[idx]
        
        if self.transform:
            features = self.transform(features, self.scalar)
            
        return features, label

def fit_scaler(train_df, features):
    """Fits the scaler on the training features only."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_df[features].values)
    return scaler

def get_dataloaders(df, batch_size=32, test_size=0.2, transform_scale=None):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    metadata = SolarFWHMetaData()

    scalar = fit_scaler(train_df, metadata.features)

    train_dataset = SolarFWH(train_df, scalar, transform_scale)
    test_dataset = SolarFWH(test_df, scalar, transform_scale)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
