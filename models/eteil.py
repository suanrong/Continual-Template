import torch
from torch.nn import functional as F
import torch.optim as optim
from basic_net import Basic_net
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import ConcatDataset

import utils
from .icarl import iCaRL
import config
config = config.config

#### Deprecated ####

class EtEIL(iCaRL):
    def _compute_loss(self, imgs, target):
        output = self.net(imgs)
        output, target = output.to(config.device), target.to(config.device)
        loss = F.cross_entropy(output, target)
        if self.old_net != None:
            old_target = self.old_net(imgs)
            for i in range(self.num_class // self.task_size - 1):
                class_idx = torch.arange(i * self.task_size, (i+1) * self.task_size).to(config.device)
                sub_output = output[..., class_idx] / config.T
                with torch.no_grad():
                    sub_target = old_target[..., class_idx] / config.T
                    sub_target = F.softmax(sub_target, 1)
                sub_output = F.log_softmax(sub_output, 1)
                loss += (-sub_target * sub_output).sum(dim=-1).mean()
        return loss


