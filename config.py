import argparse
import random

import numpy as np
import torch


def config():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.map_path = r"E:\E_downloads\Trajectory-Prediction-main (2)\data\map_files"
    args.train_path = r"E:\E_downloads\Trajectory-Prediction-main (2)\data\train\data"
    args.valid_path = r"E:\E_downloads\Trajectory-Prediction-main (2)\data\val\data"
    args.test_path = r"E:\E_downloads\Trajectory-Prediction-main (2)\data\test_obs\data"
    args.bert_num_blocks = 2
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.bert_hidden_size = 40
    args.num_heads = 2
    args.dropout = 0.2
    args.batch_size = 32
    args.hidden_size = 256

    return args
