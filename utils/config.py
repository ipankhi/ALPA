import argparse
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation of image classification with ALPA')
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--gpu_ids", type=str, required=True)
    parser.add_argument("--img_size", type=int,required=True, default=256) 

    # train or infer
    parser.add_argument("--train", type=int, required=False, default=0)
    parser.add_argument("--infer", type=int, required=False, default=0)
    parser.add_argument("--num_classes", type=int, default=5, required=False)
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--OPTIMIZER", type=str,
                        default="Adam", required=False)
    parser.add_argument("--criterion", type=str, default="CrossEntropy", required=False)
    parser.add_argument("--batchsize", type=int, default=256, required=False)
    parser.add_argument("--epochs", type=int, default=60, required=False)
    parser.add_argument("--lr", type=float, default=1e-4, required=False)
    parser.add_argument("--img_path", type=str,
                        default="/home/pankhi/long_tail/aptos_pade/DATA/APTOS2019/train_images", required=False) 
    parser.add_argument("--model_path", type=str,
                        default="chkpt", required=False)
    parser.add_argument("--store_name", type = str, help = "identify Name", default = "xXx", required = True)
    parser.add_argument("--csv_path", type=str,
                        default="/home/pankhi/long_tail/aptos_pade/DATA/APTOS2019/kfold.csv", required=False)
    parser.add_argument("--train_csv_path", type=str,
                        default="/home/pankhi/long_tail/aptos_pade/DATA/APTOS2019/train_tr.csv", required=False)
    parser.add_argument("--test_csv_path", type=str,
                        default="/home/pankhi/long_tail/aptos_pade/DATA/APTOS2019/test_tr.csv", required=False)
    parser.add_argument("--val_csv_path", type=str,
                        default="/home/pankhi/long_tail/aptos_pade/DATA/APTOS2019/val_tr.csv", required=False)
    parser.add_argument("--save_model", type=int, default=0, required=False)

    parser.add_argument("--is_master", type=int, required=False)
    
    parser.add_argument("--n_splits", type=int, required=False, default=5)

    # loss params 
    parser.add_argument("--gamma_neg",default=4, type=int, required=False)
    parser.add_argument("--gamma_pos",default=0, type=int, required=False)
    parser.add_argument("--clip",default=0.05, type=float, required=False)
    parser.add_argument("--alpha",default=0.875, type=float, required=False)
    parser.add_argument("--beta",default=1.625, type=float, required=False)
    parser.add_argument("--lamb",default=0, type=int, required=False)
    return parser.parse_args()
