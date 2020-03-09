from tensorboard_utils import Tensorboard
from MetaLearning import MetaTrainer
import argparse, random
from DataClass.torchData import *
from transformer.configuration_bart import BartConfig
from DataClass.Constants import PAD_IDX, END_IDX
from DataClass.torchData import MAX_LINE_LENGTH
import numpy as np
import torch
from torch.utils.data import DataLoader
from DataClass.MetaTorchData import MetaRepo, MetaRetrieved
from torch.multiprocessing import set_start_method
from datetime import date


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
parser.add_argument("--meta_batch_size", default=4, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--num_updates", default=10, type=int)
parser.add_argument("--k_shot", default=5, type=int)
parser.add_argument("--query_batch_size", default=10, type=int)
parser.add_argument("--total_forward", default=5, type=int)
args = parser.parse_args()

def main(args):
	random.seed(12324)
	np.random.seed(12324)
	torch.manual_seed(12324)

	VOCAB_SIZE = len(word2idx)
	num_validation_repos = 50
	tb = Tensorboard(args.exp_name, unique_name=args.unique_id)
	repo_files = list(filter(lambda x: True if x.endswith('.csv') else False, next(os.walk(args.filepath))[2]))

	print(len(repo_files))
	data_loaders = []; validation_loaders = []
	for dataset in repo_files[num_validation_repos:]:
		temp = MetaRepo(args.filepath+'/'+dataset, False, k_shot=args.k_shot, query_batch_size=args.query_batch_size)
		if len(temp) > 3: data_loaders.append(iter(DataLoader(temp, shuffle=True, batch_size=1)))
	for dataset in repo_files[:num_validation_repos]:
		temp = MetaRepo(args.filepath+'/'+dataset, False, k_shot=args.k_shot, query_batch_size=args.query_batch_size)
		if len(temp) > 3: validation_loaders.append(iter(DataLoader(temp, shuffle=True, batch_size=1)))
	# data_loaders = [iter(DataLoader(MetaRepo(args.filepath+'/'+dataset, False, k_shot=args.k_shot, query_batch_size=args.query_batch_size), shuffle=True, batch_size=1)) 
	# data_loaders = [iter(DataLoader(MetaRepo(args.filepath+'/'+dataset, False, k_shot=args.k_shot, query_batch_size=args.query_batch_size), shuffle=True, batch_size=1)) for dataset in repo_files[num_validation_repos:102]]
	# validation_data_loaders = [iter(DataLoader(MetaRepo(args.filepath+'/'+dataset, False, k_shot=args.k_shot), shuffle=True, batch_size=1)) for dataset in repo_files[98:num_validation_repos]]

	if torch.cuda.is_available:
		torch.backends.cudnn.deterministic=True
		torch.backends.cudnn.benchmark = False


	# model_params = (VOCAB_SIZE, VOCAB_SIZE, PAD_IDX, PAD_IDX, 
	# 				args.d_word_vec, args.d_word_vec, args.inner_dimension, args.num_layers,
	# 				args.num_heads, args.key_dimension, args.value_dimension, args.dropout,
	# 				MAX_LINE_LENGTH, MAX_LINE_LENGTH, True, True)

	model_params = BartConfig(vocab_size=VOCAB_SIZE, pad_token_id=PAD_IDX,
								eos_token_id=END_IDX, d_model=args.d_word_vec,
								encoder_ffn_dim=args.inner_dimension,
								encoder_layers=args.num_layers,
								encoder_attention_heads=args.num_heads,
								decoder_ffn_dim=args.inner_dimension,
								decoder_layers=args.num_layers,
								decoder_attention_heads=args.num_heads,
								dropout=args.dropout,
								max_encoder_position_embeddings=MAX_LINE_LENGTH,
								max_decoder_position_embeddings=MAX_LINE_LENGTH)
	
	device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

	trainer = MetaTrainer(args.meta_batch_size, device=device, model_params=model_params)#, total_forward=args.total_forward)
	trainer.train(data_loaders, tb, num_updates=args.num_updates)

if __name__=='__main__':
	# set_start_method('spawn', force=sFalse)
	main(args)