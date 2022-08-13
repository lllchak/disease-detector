import numpy as np

import torch
import torchvision


class ConvNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(64, 64, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(64, 128, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(128, 128, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(128, 256, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(256, 512, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLu(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True)
        )
