import os
import datetime
import argparse
import random
import numpy as np
import torch
import logging
import logging.config

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of EV-GCN')
        parser.add_argument('--train', default=0
                            , type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        parser.add_argument('--ckpt_path', type=str, default='./model', help='checkpoint path to save trained models')

        args = parser.parse_args()

        args.time = datetime.datetime.now().strftime("%y%m%d")
        args.time1 = datetime.datetime.now().strftime("%H:%M:%S")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        # self.args.printer args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")

        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        self.set_seed(0)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


