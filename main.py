from pdb import Pdb
import time
import os
from trainer import ContinualTrainer
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import numpy as np
import pandas as pd

import config
from config import setup_writer
config.init()
config = config.config
import utils

def trial(seed):
    from trainer import ContinualTrainer
    trainer = ContinualTrainer(seed)
    trainer.train()

def main():
    print(config)
    # Run several experiments with different random seed
    for seed in config.seeds:
        utils.setup_seed(seed)
        setup_writer(seed)
        start_time = time.time()
        trial(seed)
        config.logger.info("Training finished in {}".format(int(time.time() - start_time)))

if __name__ == "__main__":
    main()

    

