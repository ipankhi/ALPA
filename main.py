from train import *
# from train_random_search import *
from datasets.dataloader import *
from models.get_model import *
from models.get_optimizer import *
from utils import *
import torch
import warnings
warnings.filterwarnings("ignore")
# from libauc.metrics import auc_roc_score

if __name__ == "__main__":

    args = get_args_parser()
    init_cuda_distributed(args)
    net = get_model(args)
    if args.train:
        optimizer, criterion = make_loss_optimizer(args, net)
        train_dls, valid_dls,train_class_counts, valid_class_counts = get_dataloader(args)
        train(args, net, optimizer, criterion,train_dls, valid_dls,train_class_counts, valid_class_counts, 1)


