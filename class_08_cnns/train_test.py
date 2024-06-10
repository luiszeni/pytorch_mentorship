import lightning.pytorch as pl

import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms

from vgg import VGG16


if __name__ == '__main__':
    image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_dataset = ImageFolder('data/train', transform=image_transforms)
    test_dataset  = ImageFolder('data/test', transform=image_transforms)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=36, shuffle=True, num_workers=6, persistent_workers=True) 
    test_data_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=8, shuffle=False, num_workers=4, persistent_workers=True) 



    # for b in train_data_loader:
    print("tamanho dataset", len(train_data_loader))

    model = VGG16()


    # logger = TensorBoardLogger('tb_logs', name='image_cls_model_run', log_graph=True)



    trainer = pl.Trainer(max_epochs=2, accelerator='cpu')

    trainer.fit(model=model, train_dataloaders=train_data_loader)

    trainer.test(model, dataloaders=test_data_loader)

