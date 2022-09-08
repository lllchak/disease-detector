import torch
import model
import utils
from tqdm.autonotebook import tqdm, trange


def train(
    net: model.DeConvNet, epochs: int,
    use_gpu: bool, optimizer: torch.optim,
	scheduler: torch.optim.lr_scheduler,
	criterion: torch.nn.modules.loss, data: dict
) -> tuple:

	loss_history = {"train": [], "val": []}
	accs_history = {"train": [], "val": []}
	pbar = trange(epochs, desc="Epoch")

	for epoch in pbar:

		for phase in ["train", "val"]:
			if phase == "train":
				scheduler.step()
				net.train(True)
			else:
				net.eval()
			curr_loss = 0.0

			for batch in tqdm(data[phase], leave=False, desc=f'{phase} iter'):
				inputs, labels = batch
				if use_gpu:
					inputs, labels = inputs.cuda(), labels.cuda()

				if phase == "train":
					optimizer.zero_grad()
				if phase == "val":
					with torch.no_grad():
						outputs = net(inputs)
				else:
					outputs = net(inputs)

				loss = criterion(outputs, labels)

				if phase == "train":
					loss.backward()
					optimizer.step()

				curr_loss += loss.item()

			epoch_loss = curr_loss / 40 if phase == "train" else curr_loss / 5

			loss_history[phase].append(epoch_loss)

			pbar.set_description("{} Loss: {:.4f}".format(phase, epoch_loss))

	return net, loss_history
