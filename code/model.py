import numpy as np

import torch
import torchvision


vgg_pretrained = torchvision.models.vgg16(pretrained=True)


class ConvNet(torch.nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(64, 128, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(128, 256, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(256, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True),

            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, padding=2, return_indices=True)
        )

        self.pool_indices = {}

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(4096, n_classes)
        )

        self.init_weights()

    def init_weights(self):
        for index, layer in enumerate(vgg_pretrained.features):
            if isinstance(layer, torch.nn.Conv2d):
                self.features[index].weight.data = layer.weight.data
                self.features[index].bias.data = layer.bias.data

    def forward_conv_layers(self, value):
        res = value

        for index, layer in enumerate(self.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                pool_indx, res = layer(res)
                self.pool_indices[index] = pool_indx
            else:
                res = layer(res)
        
        return res

