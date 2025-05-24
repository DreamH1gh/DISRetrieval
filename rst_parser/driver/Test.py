import os
import sys
# sys.path.extend(["../../","../","./", "RSTparser"])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.extend([
    os.path.abspath(os.path.join(BASE_DIR, "..", "..")),
    os.path.abspath(os.path.join(BASE_DIR, ".."))
])

import argparse
import torch
import random
import numpy as np

from data.Dataloader import read_corpus, inst
from RSTParser import RSTParser

def set_random_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    set_random_seed()

    gpu = torch.cuda.is_available()
    print("GPU available:", gpu)
    print("CuDNN:", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='RSTparser/saved_model/config.cfg')
    argparser.add_argument('--use-cuda', action='store_true', default=False)
    argparser.add_argument('--test_file', default='', help='RST file')
    argparser.add_argument('--edu_path', default='')
    args = argparser.parse_args()

    print("Initializing model...")
    parser = RSTParser(args.config_file, use_cuda=(gpu and args.use_cuda))

    if args.test_file:
        test_data = read_corpus(args.test_file, args.edu_path)
        test_insts = inst(test_data)
        output_path = args.test_file + '.out'
        print(f"Running prediction and saving to {output_path}...")
        parser.predict_batch(test_insts, output_path)
