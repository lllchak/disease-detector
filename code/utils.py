import model
import torch
import matplotlib.pyplot as plt
import numpy as np


def iou_score(outputs: torch.Tensor, labels: torch.Tensor):
	SMOOTH = 1e-8
	intersection = (outputs & labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
	union = (outputs | labels).float().sum()  # Will be zzero if both are 0

	iou = (intersection + SMOOTH) / (union + SMOOTH)

	return iou


def plot_metrics(history: dict) -> None:
	plt.figure(figsize=(12, 6))

	plt.plot(history['train'])
	plt.plot(history['val'])
	plt.legend(list(history.keys()))

	plt.show()


def get_data_with_preds(
		data_type: str, data: dict,
		net: model.DeConvNet
) -> tuple:
	if data_type == "train":
		for batch in data["train"]:
			images, labels = batch
			break

	elif data_type == "val":
		for batch in data["val"]:
			images, labels = batch
			break

	elif data_type == "test":
		for batch in data["test"]:
			images, labels = batch
			break

	net.eval()

	with torch.no_grad():
		inputs = images.cuda()
		preds = net(inputs).cpu()

	return images, labels, preds


def compare_preds_with_ground_truth(
		data_type: str, data: dict,
		net: model.DeConvNet
) -> None:
	if data_type == "train":
		images, labels, preds = get_data_with_preds("train", data, net)

		plt.figure(figsize=(30, 12))

		for i in range(10):
			plt.title("prediction", fontsize=15)
			plt.subplot(3, 10, i + 1)
			plt.imshow(images[i][0].detach().numpy(), cmap="gray")

			plt.title("source", fontsize=15)
			plt.subplot(3, 10, i + 11)
			plt.imshow(labels[i][0].detach().numpy(), cmap="gray")

			plt.title("ground_truth", fontsize=15)
			plt.subplot(3, 10, i + 21)
			plt.imshow(preds[i][0].detach().numpy(), cmap="gray")

		plt.show()

	elif data_type == "val":
		images, labels, preds = get_data_with_preds("val", data, net)

		plt.figure(figsize=(30, 12))

		for i in range(10):
			plt.title("prediction", fontsize=15)
			plt.subplot(3, 10, i + 1)
			plt.imshow(images[i][0].detach().numpy(), cmap="gray")

			plt.title("source", fontsize=15)
			plt.subplot(3, 10, i + 11)
			plt.imshow(labels[i][0].detach().numpy(), cmap="gray")

			plt.title("ground_truth", fontsize=15)
			plt.subplot(3, 10, i + 21)
			plt.imshow(preds[i][0].detach().numpy(), cmap="gray")

		plt.show()

	elif data_type == "test":
		images, labels, preds = get_data_with_preds("test", data, net)

		plt.figure(figsize=(30, 12))

		for i in range(10):
			plt.title("prediction", fontsize=15)
			plt.subplot(3, 10, i + 1)
			plt.imshow(images[i][0].detach().numpy(), cmap="gray")

			plt.title("source", fontsize=15)
			plt.subplot(3, 10, i + 11)
			plt.imshow(labels[i][0].detach().numpy(), cmap="gray")

			plt.title("ground_truth", fontsize=15)
			plt.subplot(3, 10, i + 21)
			plt.imshow(preds[i][0].detach().numpy(), cmap="gray")

		plt.show()


def plot_samples(
		data: np.array,
        ground_truth: np.array
) -> None:
	plt.figure(figsize=(30, 6))

	for i in range(10):
		plt.subplot(2, 10, i + 1)
		plt.imshow(data[i], cmap="BuPu")

		plt.subplot(2, 10, i + 11)
		plt.imshow(ground_truth[i], cmap="BuPu")

	plt.show()
