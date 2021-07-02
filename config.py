import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import utils
import json
config = None
def init():
    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument('--name', type=str, default="", help='Name of experiment')
    parser.add_argument('--device', type=str, default="", help='Device ID')
    parser.add_argument('--seeds', type=int, default=[19941210, 19970801, 19650801, 19670705, 1944], nargs='+', help='a list for random seeds')
    
    parser.add_argument('--total_complexity', type=int, default=409600, help='Number of iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--bs_increment', type=int, default=0, help='The increment for batch size')
    parser.add_argument('--cifar_resnet', action='store_true', help='Use cifar version resnet')
    
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument("--scheduler", type=str, default="MultiStepLR", help="The scheduler: OneCycleLR, MultiStepLR")
    parser.add_argument("--sc", "--scheduling", default=[0.625,0.781], nargs="*", type=float, help="Where to reduce the learning rate.")
    parser.add_argument("--lr_decay", default=0.2, type=float, help="LR multiplied by it.")
    parser.add_argument("--max_lr", default=0.01, type=float, help="max LR used by 1cyclylr.")

    parser.add_argument('--eval_log', action='store_true', help='whether to eval during training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status, -1 means no log')

    parser.add_argument('--resume', default=-1, type=int, help='resume model from whitch task')
    parser.add_argument('--WTF', action='store_true', help='USE ALL GPU MEMORY')

    # specific setting
    parser.add_argument('--dataset', type=str, default="iCIFAR100", help='Dataset: iMNIST, iCIFAR100, iImageNet_Subset')
    parser.add_argument('--random_order', action='store_true', help='Use random order of classes')
    parser.add_argument('--model', type=str, default="iCaRL", help='Model')
    parser.add_argument('--task_incre', action='store_true', help='Experiment on the task incremental scenario')
    parser.add_argument('--init_task_size', type=int, default=0, help='The initial task size')
    parser.add_argument('--increment', type=int, default=10, help="the number of class in one task")
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained feature extractor and freeze it while training')
    parser.add_argument('--batch_adjust', default=-1, type=float, help="proportion for new task, -1 means no adjust, 0 means adaptive adjust")
    parser.add_argument('--finetune_on_exemplars', action='store_true', help='Finetune on exemplars before testing')

    # Exemplar
    parser.add_argument('--construct_strategy', type=str, default='herding', help='The strategy for constructing exemplar set.')
    parser.add_argument('--T', type=float, default=1., help="the temperature for distillation loss")
    parser.add_argument('--fix_memory_size', action='store_true', help='dynamic memory or fixed')
    parser.add_argument('--memory_size', type=int, default=50, help='The memory size')
    parser.add_argument('--validation_size', type=int, default=10, help='The validation set size for review')
    parser.add_argument('--classify_mode', type=str, default="CNN", help="the classify stategy of final result, CNN or NME")
    # RWN
    parser.add_argument('--review', action='store_true', help='whether to use review mechanism')
    parser.add_argument('--threshold', type=float, default=0.7, help='The recall threshold for recall')
    parser.add_argument('--visual', action='store_true', help='Visualize')
    parser.add_argument('--review_strategy', type=str, default='random_pick', help='The review strategy: random_pick, kmeans, herding')
    parser.add_argument('--access_limit', default=-1, type=int, help='The access times limit')
    parser.add_argument('--ce', action='store_true', help='Use ce loss instead of bce')
    # ucir
    parser.add_argument('--mimic_score', action='store_true', help='Use mimic score loss')
    parser.add_argument('--less_forget', action='store_true', help='Use less forget loss')
    parser.add_argument('--margin_rank', action='store_true', help='Use margin rank loss')
    # ewc 
    parser.add_argument('--ewc_lambda', type=float, default=40, help='The recall threshold for recall')

    parser.add_argument('--DJB', default=0.0, type=float, help='The DJB')
    # setup device logger, tensorboard
    global config
    config = parser.parse_args()
    if config.device != "":
        deviceID = config.device
    else:
        deviceID = str(utils.get_free_gpu())
    config.device = 'cuda:'+deviceID if torch.cuda.is_available() else 'cpu'
    if config.WTF:
        utils.occumpy_mem(deviceID)
    from datetime import datetime
    current_time = datetime.now().strftime('%b_%d,%H:%M:%S')
    log_dir = "runs/" + config.name + "," + current_time
    import os
    os.mkdir(log_dir)
    config.log_dir = log_dir
    with open(log_dir + '/config.txt', 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    config.logger = utils.setup_logger(log_dir)
    config.logger.info("Use device {}".format(config.device))

def setup_writer(seed):
    global config
    config.writer = SummaryWriter(config.log_dir + "/" + str(seed))
    # general tools
    
    
    
