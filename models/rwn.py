import torch
from torch.nn import functional as F
import torch.nn as nn

from .icarl import iCaRL
from basic_net import Basic_net
import utils
import config
config = config.config


class AngleLayer(nn.Linear):
    """Convert the fully connected layer of output to """
    def __init__(self, in_features, out_features, bias=False):
        super(AngleLayer, self).__init__(in_features, out_features, bias)
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        w = F.normalize(self.weight)
        return F.linear(x, w, self.bias)

class RWN(iCaRL):
    def _construct_basic_net(self):
        '''
           may need to overwrite this func if you want to use a new NN. 
        '''
        basic_net = Basic_net()
        basic_net.fc_module = AngleLayer
        basic_net.bias = False
        return basic_net

    def _compute_loss(self, imgs, target):
        output = self.net(imgs)
        tt = target
        loss = 0
        # for param in self.net.fc.parameters():
        #     loss += 0.000001 * torch.norm(param, 1)**2
        if self.old_net == None:
            if config.ce:
                return loss + F.cross_entropy(output, target)
            else:
                target = utils.to_one_hot(target, self.num_class).to(config.device)
                return loss + F.binary_cross_entropy_with_logits(output, target)
        else:
            target = utils.to_one_hot(target, self.num_class).to(config.device)
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_net(imgs))
            old_task_size = old_target.shape[1]
            idx = torch.nonzero(tt >= old_task_size).flatten()
            target[idx, :old_task_size] = old_target[idx]

            if config.ce:
                log_prob = F.log_softmax(output, dim=-1)
                return loss + (-target * log_prob).sum(dim=-1).mean()
                ## Not implement: reduction != mean 
            return loss + F.binary_cross_entropy_with_logits(output, target)


