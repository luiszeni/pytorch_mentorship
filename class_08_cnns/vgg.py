

import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as pl

import torch.utils.model_zoo as model_zoo


class VGG16(pl.LightningModule):

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_features = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )

        self.cls_head = nn.Linear(4096, num_classes)

    

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear_features(x)
        x = self.cls_head(x)
        return x
    

    def training_step(self, batch):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        # Computes ACC
        cls_result = y_hat.argmax(dim=1)
        correct = cls_result.eq(y).sum().item()
        acc = correct/len(y)

        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.log("acc", acc, prog_bar=True, on_epoch=True)
        
        return loss
    

    def test_step(self, batch, batch_idx):
        # test_step defines the test loop.
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        # Computes ACC
        cls_result = y_hat.argmax(dim=1)
        correct = cls_result.eq(y).sum().item()
        acc = correct/len(y)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer