import os
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from football_dataset_classification import footballDataset
import torchvision.models as models
from utils import collate_fn



class myModel(nn.Module):
    def __init__(self, num_jerseys=10, num_colors=2):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=16)
        self.conv2 = self.make_block(in_channels=16, out_channels=32)
        self.conv3 = self.make_block(in_channels=32, out_channels=64)
        self.conv4 = self.make_block(in_channels=64, out_channels=64)
        self.conv5 = self.make_block(in_channels=64, out_channels=64)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=3136, out_features=1024),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
        )
        self.fc3_1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_jerseys),
        )
        self.fc3_2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_colors),
        )

    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x1 = self.fc3_1(x)
        x2 = self.fc3_2(x)

        return x1, x2


# Transfer learning
class resnet50_two_header(nn.Module):
    def __init__(self, num_jersey=10, num_color=2):
        super().__init__()

        self.model = models.resnet50(pretrained=True)
        self.model.fc1 = nn.Linear(in_features=2048, out_features=num_jersey)
        self.model.fc2 = nn.Linear(in_features=2048, out_features=num_color)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x1 = self.model.fc1(x)
        x2 = self.model.fc2(x)

        return x1, x2



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])

    path = "dataset/football_train"
    dataset = footballDataset(root=path, transform=transform)

    params = {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 6,
        "drop_last": True,
        "collate_fn": collate_fn,
    }
    dataloader = DataLoader(dataset, **params)

    # model = myModel(10, 2).to(device)
    model = resnet50_two_header(10,2).to(device)

    for images, labels, colors in dataloader:
        images = images.to(device)

        jersey_predict, color_predict = model(images)
        print(jersey_predict.shape)
        print(color_predict.shape)
        print("---------------")
