import torch
from torch.nn import functional as F
import copy

from .replay import Replay
import utils
import config
config = config.config

class iCaRL(Replay):
    """
        This is the base class for method using NME
    """
    def __init__(self, train_dataset, test_dataset, *args):
        super().__init__(train_dataset, test_dataset, *args)
        self.old_net = None

    def after_train(self, task):
        super().after_train(task)
        del self.old_net
        self.old_net = copy.deepcopy(self.net)
        self.old_net.to(config.device)
        self.old_net.eval()

    def _compute_loss(self, imgs, target):
        output = self.net(imgs)
        target = utils.to_one_hot(target, self.num_class).to(config.device)
        if self.old_net == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            with torch.no_grad():
                old_target = torch.sigmoid(self.old_net(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)

# class iCaRL_new(Replay):
#     """
#         This is the base class for method using NME:::   multi distillation
#     """
#     def __init__(self, train_dataset, test_dataset, *args):
#         super().__init__(train_dataset, test_dataset, *args)
#         self.old_net = None

#     def _after_train(self, task):
#         super()._after_train(task)
#         old_net = copy.deepcopy(self.net)
#         old_net.to(config.device)
#         old_net.eval()
#         if not self.old_net:
#             self.old_net = []
#         self.old_net.append(old_net)
        
#     def _compute_loss(self, imgs, target):
#         output = self.net(imgs)
#         target = utils.to_one_hot(target, self.num_class).to(config.device)
#         if self.old_net == None:
#             return F.binary_cross_entropy_with_logits(output, target)
#         else:
#             with torch.no_grad():
#                 old_target = []
#                 for i, old_net in enumerate(self.old_net):
#                     old_target.append(torch.sigmoid(old_net(imgs))[..., i*config.increment:(i+1)*config.increment])
#             old_target = torch.hstack(old_target)
#             old_task_size = old_target.shape[1]
#             target[..., :old_task_size] = old_target
#             return F.binary_cross_entropy_with_logits(output, target)

