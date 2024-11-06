import os
import shutil
import torch
import torch.nn as nn
from pandas.io.sas.sas_constants import dataset_offset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from football_dataset_classification import footballDataset
from utils import plot_confusion_matrix, get_arg, collate_fn


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])

    train_set = footballDataset(root=args.data_path, transform=transform)

    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 6,
        "drop_last": True,
        "collate_fn": collate_fn,
    }
    valid_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 6,
        "drop_last": False,
        "collate_fn": collate_fn,
    }
    dataloader = DataLoader(train_set, **training_params)



if __name__ == '__main__':
    args = get_arg()
    train(args)




