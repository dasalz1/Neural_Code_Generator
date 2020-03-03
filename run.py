from tensorboard_utils import Tensorboard
from transformer.Models import Transformer
from DataClass.torchData import *
from DataClass.Constants import PAD_IDX
from DataClass.torchData import MAX_LINE_LENGTH
import torch.optim as optim
from EditorNoRet import EditorNoRetrievalTrainer
from torch.utils.data import ConcatDataset, DataLoader
import torch
from datetime import date
import argparse, random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--filepath", default='../repo_files', type=str)
parser.add_argument("--exp_name", default='EditorPairTrain', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--key_dimension", default=64, type=int)
parser.add_argument("--value_dimension", default=64, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--d_word_vec", default=512, type=int)
parser.add_argument("--inner_dimension", default=2048, type=int)
parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--epochs", default=10, type=int)
args = parser.parse_args()

def main(args):
	random.seed(12324)
	np.random.seed(12324)
	torch.manual_seed(12324)

	VOCAB_SIZE = len(word2idx)
	num_validation_repos = 100

	tb = Tensorboard(args.exp_name, unique_name=args.unique_id)

	repo_files = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(args.filepath))[2]))

	data_loader = DataLoader(ConcatDataset([PairDataset(args.filepath +'/'+dataset) for dataset in repo_files[num_validation_repos:]]),
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=batch_collate_fn,
							num_workers=16)

	print("Finished creating data loader")
	# data_loader = DataLoader(PairDataset(args.filepath+'/'+repo_files[30]), batch_size=args.batch_size, shuffle=True, collate_fn=batch_collate_fn)

	validation_loader = DataLoader(ConcatDataset([PairDataset(args.filepath +'/'+dataset) for dataset in repo_files[:num_validation_repos]]),
							batch_size=args.batch_size,
							shuffle=True,
							collate_fn=batch_collate_fn,
							num_workers=16)

	print("Finished creating validation data loader")
	num_iterations = len(data_loader)

	model = Transformer(n_src_vocab=VOCAB_SIZE, n_trg_vocab=VOCAB_SIZE, src_pad_idx=PAD_IDX, trg_pad_idx=PAD_IDX, 
						d_word_vec=args.d_word_vec, d_model=args.d_word_vec, d_inner=args.inner_dimension, n_layers=args.num_layers,
						n_head=args.num_heads, d_k=args.key_dimension, d_v=args.value_dimension, dropout=args.dropout,
						n_trg_position=MAX_LINE_LENGTH, n_src_position=MAX_LINE_LENGTH, 
						trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if torch.cuda.is_available:
		torch.backends.cudnn.deterministic=True
		torch.backends.cudnn.benchmark = False
		
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs...")
		model = torch.nn.DataParallel(model)
	
	# model.to(device)

	src_word_emb
	
	trainer = EditorNoRetrievalTrainer(device)
	optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.995), eps=1e-8)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.25 * num_iterations), round(0.5 * num_iterations), round(0.75 * num_iterations)], gamma=0.1)

	trainer.train(model, optimizer, data_loader, validation_loader, tb=tb, epochs=args.epochs)

if __name__=='__main__':
	main(args)