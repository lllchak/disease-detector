import torch
import model
from tqdm.autonotebook import tqdm, trange


def train(net: model.DeConvNet, epochs: int, use_gpu: bool, scheduler, optimizer, criterion, data) -> tuple:
	net.train(True)

	loss_history = []
	pbar = trange(epochs, desc="Epoch")

	for epoch in pbar:
		scheduler.step()

		curr_loss = 0.0

		for batch in data:
			inputs, labels = batch
			if not use_gpu:
				inputs, labels = inputs.cuda(), labels.cuda()

			optimizer.zero_grad()
			outputs = net(inputs)

			loss = criterion(outputs, labels)

			loss.backward()
			optimizer.step()

			curr_loss += loss.item()

		epoch_loss = curr_loss / 100

		loss_history.append(epoch_loss)

		pbar.set_description("Epoch: {} Loss: {:.4f}".format(epoch, epoch_loss))

	return net, loss_history

