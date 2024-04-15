
from torch.utils.data.dataset import Dataset

class HeartDeseaseDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.x = [0, 1, 4, 6, 7, 7 ,5, 3]
        self.y = [0, 0, 1, 1, 0, 0 ,1, 0]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):

        return self.x[idx], self.y[idx]