
import os
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class HeartDeseaseDataset(Dataset):

    def __init__(self, dataset_location, train=True):
        super().__init__()

        csv_path = 'train.csv' if train else 'test.csv'
        csv_path = os.path.join(dataset_location, csv_path)

        self.dataset_df = pd.read_csv(csv_path)

        self.n_of_features = len(self.dataset_df.columns) - 2
        self.n_of_classes  = int(self.dataset_df['target'].max() + 1)
   
    
    def __len__(self):
        return len(self.dataset_df)
    

    def __getitem__(self, idx):
        data_sample = self.dataset_df.iloc[idx]

        x_cols = list(data_sample.keys())
        del x_cols[0]
        del x_cols[x_cols.index('target')]

        x = data_sample[x_cols]
        y = data_sample['target']

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        
        return x, y