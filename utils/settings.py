import os
import random
import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelPrecision
# from libauc.metrics import auc_roc_score
from sklearn.metrics import roc_auc_score as auc_roc_score

def init_cuda_distributed(args):
    try:
        if torch.cuda.is_available():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
            torch.distributed.init_process_group(
                backend='nccl', init_method='env://')
            args.local_rank = torch.distributed.get_rank()
            args.world_size = torch.distributed.get_world_size()
            args.is_master = args.local_rank == 0
            args.device = torch.device(
                f'cuda:{args.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
            torch.cuda.set_device(args.local_rank)
            seed_everything(args.seed + args.local_rank)
        else:
            args.local_rank = 0
            args.world_size = 1
            args.is_master = True
            args.device = torch.device('cpu')
            seed_everything(args.seed)
    except Exception as e:
        print(f"Error during CUDA initialization: {e}")

# Set Seed


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False


def metrics(args, pred, label):
    metric_precision = MultilabelPrecision(num_labels=args.num_classes, average="macro")
    print(pred.shape, label.shape)
    precision = metric_precision(pred, label)
    auc_score = np.mean(auc_roc_score(label, pred))

    pred = pred > 0.5
    acc = (pred == label).float().mean()

    return precision, acc, auc_score


def save_model(args, model, idx=None):
    # Ensure the model path directory exists
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Construct the model file name
    model_file_name = f"{args.model}_{args.img_size}_{args.batchsize}{args.store_name}.pt"
    
    # Save the model's state dictionary
    torch.save(model.state_dict(), os.path.join(args.model_path, model_file_name))
    
    print(f"Model saved at {os.path.join(args.model_path, model_file_name)}")
