import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from .icarl import iCaRL

import utils
import config
config = config.config


class BiasCorrectionLayer(nn.Linear):
    """Convert the fully connected layer of output to """
    def __init__(self, fc, num_class):
        super().__init__(fc.in_features, fc.out_features)
        self.weight_old = Parameter(torch.Tensor(num_class - config.increment, fc.in_features))
        self.weight_new = Parameter(torch.Tensor(config.increment, fc.in_features))
        self.weight_old.data = fc.weight.data[:-config.increment]
        self.weight_new.data = fc.weight.data[-config.increment:]
        self.bias_old = Parameter(torch.Tensor(num_class - config.increment))
        self.bias_new = Parameter(torch.Tensor(config.increment))
        self.bias_old.data = fc.bias.data[:-config.increment]
        self.bias_new.data = fc.bias.data[-config.increment:]
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        old_logit = F.linear(x, self.weight_old, self.bias_old)
        new_logit = self.alpha * F.linear(x, self.weight_new, self.bias_new) + self.beta
        return torch.cat([old_logit, new_logit],1)

class BiC(iCaRL):
    """
        Large Scale Incremental Learning CVPR'19
    """        
    def _compute_loss(self, imgs, target):
        output = self.net(imgs)
        loss = F.cross_entropy(output, target)
        target = utils.to_one_hot(target, self.num_class).to(config.device)
        if self.old_net != None:
            with torch.no_grad():
                old_target = self.old_net(imgs)
            old_target = F.softmax(old_target / config.T, 1)
            output = F.log_softmax(output[..., :-config.increment] / config.T, 1)
            loss += (-old_target * output).sum(dim=-1).mean()
        return loss

    def after_train(self, task):
        super().after_train(task)
        if (task > 0):
            test_loader = self.test_dataset.get_loader(task)
            old_acc, _ = self._test(test_loader)
            self.net.fc = BiasCorrectionLayer(self.net.fc, self.num_class).to(config.device)
            validation_dataset = self.exemplar_handler.get_validation_set()
            dataloader = DataLoader(dataset=validation_dataset, batch_size=config.batch_size, num_workers=4)
            optim = getattr(torch.optim, config.optim)((self.net.fc.alpha, self.net.fc.beta), lr=0.01, weight_decay=config.weight_decay)
            prog_bar = tqdm(range(20))
            for epoch in prog_bar:
                loss_sum = 0.0
                for step, (images, target) in enumerate(dataloader):
                    images, target = images.to(config.device), target.to(config.device)
                    output = self.net(images)
                    loss = F.cross_entropy(output, target)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    loss_sum += loss.item()
                accuracy, _ = self._test(test_loader)
                prog_bar.set_description(
                "Task {} bias-tuning -> alpha {} beta {} => loss: {} test accuracy : {}".format(
                    task, 
                    round(self.net.fc.alpha.item(),3), round(self.net.fc.beta.item(),3),
                    loss_sum / (step + 1),
                    accuracy
                ))
            config.logger.info("alpha {} beta {} accuracy {} ->> {}".format(round(self.net.fc.alpha.item(),3), round(self.net.fc.beta.item(),3), old_acc, accuracy))


