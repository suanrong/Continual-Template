import torch
from torch.nn import functional as F
from torch.autograd import Variable

from .base import Base
import config
config = config.config

class EWC(Base):
    tasks=[]
    def _after_train(self, task):
        super()._after_train(task)
        self.consolidate(self.estimate_fisher(task), task)
    
    def _compute_loss(self, imgs, target):
        loss1 = super()._compute_loss(imgs, target)
        loss2 = self._ewc_loss()
        return loss1 + loss2
        
    def _ewc_loss(self):
        if len(self.tasks) > 0:
            losses = []
            for task in self.tasks:
                for n, p in self._get_named_parameters():
                    n = n.replace('.', '__')
                    mean = getattr(self.net, '{}_mean_{}'.format(n,task))
                    fisher = getattr(self.net, '{}_fisher_{}'.format(n,task))
                    mean = Variable(mean)
                    fisher = Variable(fisher)
                    loss = (fisher * (p-mean)**2).sum()
                    losses.append(loss)
            return (config.ewc_lambda / 2) * sum(losses)
        else:
            return 0

    def estimate_fisher(self, task):
        data_loader = self.train_dataset.get_loader(task)
        # init 
        est_fisher_info = {}
        for n, p in self._get_named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()
        # cal
        for step, (images, target) in enumerate(data_loader):
            images, target = images.to(config.device), target.to(config.device)
            loglikelihoods = F.log_softmax(self.net(images), dim=1)
            negloglikelihood = F.nll_loss(loglikelihoods, target)
            self.net.zero_grad()
            negloglikelihood.backward()
            for n, p in self._get_named_parameters():
                n = n.replace('.', '__')
                est_fisher_info[n] += p.grad.detach() ** 2
        # mean
        est_fisher_info = {n: p / len(data_loader) for n, p in est_fisher_info.items()}
        return est_fisher_info

    def _get_named_parameters(self):
        return [[n,p] for n,p in self.net.named_parameters() if p.requires_grad and n!="fc.weight" and n!="fc.bias"]

    def consolidate(self, fisher, task):
        self.tasks.append(task)
        print(self.tasks)
        for n, p in self._get_named_parameters():
            n = n.replace('.', '__')
            self.net.register_buffer('{}_mean_{}'.format(n,task), p.data.clone())
            self.net.register_buffer('{}_fisher_{}'.format(n,task), fisher[n].data.clone())
    