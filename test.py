from configs import *
import os
import sys
from os import mkdir

import torch
from torch import nn
from torch.backends import cudnn

sys.path.append('.')
from data import make_data_loader
from inference import inference
from modeling import build_model
from logger import setup_logger


def main():
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    output_dir = OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))

    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader()

    model = build_model(num_classes) #
    # import pdb
    # pdb.set_trace()
    model.load_state_dict(torch.load(TEST_WEIGHT)["model"])
    model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(TEST_WEIGHT))

    #total_loader = torch.utils.data.ConcatDataset([train_loader, val_loader])

    inference(model, val_loader, num_query)


if __name__ == '__main__':
    main()
