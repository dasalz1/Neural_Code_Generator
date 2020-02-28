import torch
import torch.nn as nn
import torch.nn.functional as F
from DataClass.Constants import PAD_IDX
from train_utils import save_checkpoint, from_checkpoint_if_exists, tb_mle_epoch, tb_mle_batch
from tqdm import tqdm

class EditorNoRetrievalTrainer:

	def __init__(self, device):
		self.device = device

	def compute_mle_loss(self, pred, target, smoothing, log=False):
		def compute_loss(pred, target, smoothing):
			target = target.contiguous().view(-1)
			if smoothing:
			  eps = 0.1
			  n_class = pred.size(1)

			  one_hot = torch.zeros_like(pred)
			  one_hot = one_hot.scatter(1, target.view(-1, 1), 1)
			  one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
			  log_prb = F.log_softmax(pred, dim=1)

			  non_pad_mask = target.ne(PAD_IDX)
			  loss = -(one_hot * log_prb).sum(dim=1)
			  loss = loss.masked_select(non_pad_mask).sum()  # average later
			else:
				
			  loss = F.cross_entropy(pred, target, ignore_index=PAD_IDX, reduction='sum')    
			return loss
		
		loss = compute_loss(pred, target, smoothing)
		pred_max = pred.max(1)[1]
		target = target.contiguous().view(-1)
		non_pad_mask = target.ne(PAD_IDX)
		n_correct = pred_max.eq(target)
		n_correct = n_correct.masked_select(non_pad_mask).sum().item()

		return loss, n_correct


	def train(self, model, optimizer, data_loader, scheduler=None, tb=None, epochs=20, log_interval=100, checkpoint_interval=10000):
		
		curr_epoch, model, optimizer, scheduler = from_checkpoint_if_exists(model, optimizer, scheduler)
		model.train()

		for epoch in range(epochs):
			total_mle_loss = 0.0
			n_word_total = 0.0
			n_word_correct = 0.0
			for batch_idx, batch in enumerate(tqdm(data_loader, mininterval=2, leave=False)):
				batch_xs, batch_ys = map(lambda x: x.to(self.device), batch)
				trg_ys = batch_ys[:, 1:]
				optimizer.zero_grad()

				pred_logits = model(batch_xs, batch_ys[:, :-1])
				pred_logits = pred_logits.view(-1, pred_logits.size(2))
				loss, n_correct = self.compute_mle_loss(pred_logits, trg_ys, smoothing=True)

				loss.backward()
				optimizer.step()

				if scheduler:
					scheduler.step()

				total_mle_loss += loss.item()

				non_pad_mask = trg_ys.ne(PAD_IDX)
				n_word = non_pad_mask.sum().item()
				n_word_total += n_word
				n_word_correct += n_correct

				if tb is not None and batch_idx % log_interval == 0:
					tb_mle_batch(tb, total_mle_loss, n_word_total, n_word_correct, epoch, batch_idx, len(data_loader))

				if batch_idx != 0 and batch_idx % checkpoint_interval == 0:
					save_checkpoint(epoch, model, optimizer, scheduler, suffix=str(batch_idx))
			
			loss_per_word = total_loss / n_word_total
			accuracy = n_word_correct / n_word_total

			if tb is not None:
				tb_mle_epoch(tb, loss_per_word, accuracy, epoch)

