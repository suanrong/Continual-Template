import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.nn import Parameter
from torchvision.models.resnet import BasicBlock
import math

from .icarl import iCaRL
from basic_net import Basic_net
import config
config = config.config


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=True, *args, **kargs):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1) #for initializaiton of sigma

    def forward(self, x):
        out = F.linear(F.normalize(x, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

class Last_basicblock(BasicBlock):
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        ## without relu
        return out

class UCIR(iCaRL):
    def _construct_basic_net(self):
        '''
           may need to overwrite this func if you want to use a new NN. 
        '''
        basic_net = Basic_net()
        basic_net.fc_module = CosineLinear
        basic_net.bias = False
        if config.cifar_resnet:
            basic_net.feature.layer3[-1] = Last_basicblock(64, 64)
        else:
            basic_net.feature.layer4[-1] = Last_basicblock(512, 512)
        return basic_net

    def _compute_loss(self, imgs, target):
        output = self.net(imgs)
        output, target = output.to(config.device), target.to(config.device)
        loss = F.cross_entropy(output, target)
        if self.old_net != None:
            if config.less_forget:
                with torch.no_grad():
                    old_features = self.old_net.extract_feature(imgs)
                    features = self.net.extract_feature(imgs)
                lambda_base = 5
                scheduled_lambda = lambda_base * math.sqrt((self.num_class - config.increment) / config.increment)
                disG_loss = F.cosine_embedding_loss(old_features, features, torch.ones(features.shape[0]).to(config.device))
                loss += scheduled_lambda * disG_loss
            elif config.mimic_score:
                with torch.no_grad():
                    old_output = self.old_net(imgs)
                    old_scores = old_output / self.old_net.fc.sigma
                    ref_scores = output[...,:old_scores.shape[1]] / self.net.fc.sigma
                loss += nn.MSELoss()(old_scores, ref_scores) * (self.num_class - config.increment)
            if config.margin_rank:
                ranking_loss = self.ucir_ranking_loss(output, target)
                loss += ranking_loss
        return loss

    def ucir_ranking_loss(self, logits, targets):
        K = 2
        margin = 0.5
        gt_index = torch.zeros(logits.size()).to(logits.device)
        gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
        gt_scores = logits.masked_select(gt_index)
        # get top-K scores on novel classes
        num_old_classes = self.num_class - config.increment
        max_novel_scores = logits[:, num_old_classes:].topk(K, dim=1)[0]
        # the index of hard samples, i.e., samples of old classes
        hard_index = targets.lt(num_old_classes)
        hard_num = torch.nonzero(hard_index).size(0)

        if hard_num > 0:
            gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
            max_novel_scores = max_novel_scores[hard_index]
            assert (gt_scores.size() == max_novel_scores.size())
            assert (gt_scores.size(0) == hard_num)
            loss = nn.MarginRankingLoss(margin=margin)(gt_scores.view(-1, 1), \
                max_novel_scores.view(-1, 1), torch.ones(hard_num*K).to(config.device))
            return loss
        return 0
