from data_argument import utils, dataloader
import torch
import numpy as np
import time

from model.graph.CSA4Rec import CSA4Rec
from parse import parse_args
import multiprocessing
import os
from data_argument.trainers import GraphRecTrainer
from data_argument.utils import EarlyStopping
from data.loader import FileIO
from util.conf import ModelConf
from warnings import simplefilter

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
args.cores = multiprocessing.cpu_count() // 2

utils.set_seed(args.seed)


simplefilter(action="ignore", category=FutureWarning)
dataset = dataloader.Loader(args)
if args.model_name == 'CSA4Rec':
    conf = ModelConf('./conf/' + args.model_name + '.conf')
    conf.config['training.set'] = './dataset/' + args.data_name + '/train.txt'
    conf.config['test.set'] = './dataset/' + args.data_name + '/test.txt'
    conf.config['uu'] = args.uu
    conf.config['ii'] = args.ii
    conf.config['dataname'] = args.data_name
    training_data = FileIO.load_data_set(conf['training.set'], conf['model.type'])
    test_data = FileIO.load_data_set(conf['test.set'], conf['model.type'])
    model = CSA4Rec(conf, training_data, test_data, True, args).model
model = model.to(args.device)
trainer = GraphRecTrainer(model, dataset, args)
checkpoint_path = 'checkpoints/'+args.data_name+'.pth'
print(f"load and save to {checkpoint_path}")
print(f'Load model from {checkpoint_path} for test!')
trainer.load(checkpoint_path)

[distill_row, distill_col, distill_val] = trainer.generateKorderGraph(userK=args.distill_userK, itemK=args.distill_itemK, threshold=args.distill_thres)
dataset.reset_graph([distill_row, distill_col, distill_val])
model.dataset = dataset
model.reset_all()
model.n_layers = args.distill_layers
trainer.optim = torch.optim.Adam(model.parameters(), lr=args.lr)
trainer.dataset = dataset
checkpoint_path = utils.getDistillFileName("./checkpoints_distill/", args)
early_stopping = EarlyStopping(checkpoint_path, patience=10, verbose=True)

s = time.time()
for epoch in range(args.epochs):
    trainer.train(epoch)
    if (epoch+1) %10==0:
        scores, _, _ = trainer.valid(epoch, full_sort=True)
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
print('---------------Change to Final testing!-------------------')
trainer.model.load_state_dict(torch.load(checkpoint_path))
valid_scores, _, _ = trainer.valid('best', full_sort=True)
scores, result_info, _ = trainer.complicated_eval()
e = time.time()
print("Running time: %f s" % (e - s))