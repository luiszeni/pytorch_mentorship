from torch import nn

class MPModel(nn.Module):

    def __init__(self, in_size=30, n_classes=2):
        super(MPModel, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.cls_head(x)


