import torch
import torchvision

vgg_pretrained = torchvision.models.vgg16(pretrained=True)


class ConvNet(torch.nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()

		self.features = torch.nn.Sequential(
			torch.nn.Conv2d(3, 64, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(64, 64, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, stride=2, return_indices=True),

			torch.nn.Conv2d(64, 128, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(128, 128, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, stride=2, return_indices=True),

			torch.nn.Conv2d(128, 256, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(256, 256, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(256, 256, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, stride=2, return_indices=True),

			torch.nn.Conv2d(256, 512, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(512, 512, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(512, 512, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, stride=2, return_indices=True),

			torch.nn.Conv2d(512, 512, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(512, 512, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.Conv2d(512, 512, (3, 3), padding=1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2, stride=2, return_indices=True),

			torch.nn.Conv2d(512, 4096, (8, 8), padding=0),
			torch.nn.Conv2d(4096, 4096, (1, 1), padding=0)
		)

		self.pool_indices = {}

		self.init_weights()

	def init_weights(self) -> None:
		for index, layer in enumerate(vgg_pretrained.features):
			if isinstance(layer, torch.nn.Conv2d):
				self.features[index].weight.data = layer.weight.data
				self.features[index].bias.data = layer.bias.data

	def forward_conv_layers(self, value: torch.Tensor) -> torch.Tensor:
		res = value

		for index, layer in enumerate(self.features):
			if isinstance(layer, torch.nn.MaxPool2d):
				res, pool_indx = layer(res)
				self.pool_indices[index] = pool_indx
			else:
				res = layer(res)

		return res

	def forward(self, value: torch.Tensor) -> list:
		res = self.forward_conv_layers(value)

		return res
