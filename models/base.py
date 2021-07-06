import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
from basic_net import Basic_net, DJB
import utils
import config
config = config.config

from typing import final


class Base:
    '''
        This is a base class for continual learning.
    '''
    def __init__(self, train_dataset, test_dataset, pretrained, *args):
        self.net = self._construct_basic_net()
        self.num_class = 0
        self.name = config.name
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        if pretrained:
            self._pretrain()

    def expand_net(self, num_class):
        if num_class > self.num_class:
            self.num_class = num_class
            self.net.expand_fc(self.num_class)
        self.net.to(config.device)

    def setup_optimizer(self, optim=None, scheduler=None):
        ## TODO xxx
        if not optim:
            optim = config.optim
        if not scheduler:
            scheduler = config.scheduler
        if optim == "Adam":
            self.optim = getattr(torch.optim, optim)(filter(lambda p: p.requires_grad, self.net.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        else:
            self.optim = getattr(torch.optim, optim)(filter(lambda p: p.requires_grad, self.net.parameters()), lr=config.lr, weight_decay=config.weight_decay, momentum=config.momentum)
        
        if scheduler == "MultiStepLR":
            milestones = [int(x * config.total_complexity / config.batch_size) for x in config.sc]
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(self.optim, milestones=milestones, gamma=config.lr_decay)
        else:
            self.scheduler = getattr(torch.optim.lr_scheduler, scheduler)(self.optim, max_lr=config.max_lr, total_steps=int(config.total_complexity / config.batch_size))
        
    def save_model(self, task):
        ##### TODO ######
        # for m in self.net.feature.children():
        #     if isinstance(m, DJB):
        #         m.save_para()
        # self.net.fc.save_para()
        #################
        self.net.eval()
        filename='model/{}_task:{}_seed:{}.pt'.format(self.name, task, self.train_dataset.seed)
        torch.save(self.net, filename)

    def classify(self, imgs, task):
        self.net.eval()
        outputs = self.net(imgs)
        if config.task_incre:
            st = self.train_dataset.class_split[task]
            en = self.train_dataset.class_split[task+1]
            return torch.argmax(outputs[:,st:en], 1) + self.train_dataset.class_split[task]
        return torch.argmax(outputs, dim=1)

    def resume(self, task, name=None):
        if not name and (config.resume < 0 or task > config.resume):
            return False
        if not name:
            name = config.name
        filename='model/{}_task:{}_seed:{}.pt'.format(name, task, self.train_dataset.seed)    
        if os.path.isfile(filename):
            try:
                # self.net.load_state_dict(torch.load(filename))
                self.net = torch.load(filename)
                self.net.to(config.device)
                self.net.eval()
                config.logger.info("Load success!")
                return True
            except:
                pass
        
        config.logger.info("Load {} failed!".format(filename))
        return False

    def _compute_all_loss(self, images, target, all_loss):
        class_split = self.train_dataset.class_split
        for i in range(len(class_split) - 1):
            st_class = class_split[i]
            en_class = class_split[i+1]
            if st_class > torch.max(target):
                break
            idxes = torch.where(torch.logical_and(target >= st_class, target < en_class))[0]
            if len(idxes) > 0:
                loss = self._compute_loss(images[idxes], target[idxes])
                label = "separate loss {}-{}".format(
                    str(st_class).rjust(2, "0"),
                    str(en_class - 1).rjust(2, "0"))
                if label not in all_loss:
                    all_loss[label] = [loss]
                else:
                    all_loss[label].append(loss)

    @final
    def train(self, task, train_loader, valid_loader=None):
        accuracy = None
        config.logger.info("Data size : {}".format(len(train_loader.dataset)))
        iterloader = iter(train_loader)
        loss = 0.0
        num_iters = int(config.total_complexity / config.batch_size)
        prog_bar = tqdm(range(num_iters))
        for step in prog_bar:
            self.net.train()
            try:
                images, target = next(iterloader)
            except StopIteration:
                iterloader = iter(train_loader)
                images, target = next(iterloader)
            
            images, target = images.to(config.device), target.to(config.device)
            loss_value = self._compute_loss(images, target)
            self.optim.zero_grad()
            loss_value.backward()
            self.optim.step()
            self.scheduler.step()
            loss += loss_value.item()
            if config.log_interval > 0 and (step + 1) % config.log_interval == 0:
                if config.eval_log:
                    accuracy, _, conf_m = self.eval(task)
                    config.writer.add_scalar("train_" + str(task) + "_accuracy", accuracy, global_step=step)
                    config.writer.add_figure('confusion_matrix_{}'.format(task), conf_m, global_step=step)
                loss /= config.log_interval
                prog_bar.set_description(
                "Task {} Iteration {}/{} lr {} => loss: {} Test accuracy : {}%".format(
                    task, 
                    step + 1, num_iters,
                    round(self.scheduler.get_last_lr()[0], 10),
                    round(loss, 5),
                    accuracy
                ))
                if valid_loader:
                    early_stopping = self._validation(valid_loader, loss)
                    if early_stopping:
                        config.logger.info("early_stopping")
                        break
                config.writer.add_scalar("train_" + str(task) + "_loss", loss / config.log_interval, global_step=step)
                loss = 0.0

    def _validation(self, valid_loader, train_loss):
        self.net.eval()
        loss = 0
        correct, total = 0, 0
        pred = []
        y_true = []
        with torch.no_grad():
            for step, (images, target) in enumerate(valid_loader):
                images, target = images.to(config.device), target.to(config.device)
                predicts = self.classify(images)
                pred.append(predicts)
                y_true.append(target)

                loss_value = self._compute_loss(images, target)
                loss += loss_value.item()
        correct = (torch.cat(pred) == torch.cat(y_true))
        accuracy = round(100.0 * correct.sum().item() / len(torch.cat(pred)), 3)
        loss = loss / (step + 1)
        config.logger.info("valid loss: " + str(loss))
        config.logger.info("valid accuracy: " + str(accuracy))
        self.net.train()
        return False

    def after_train(self, task):
        pass 

    def test(self, testloader, task):
        self.net.eval()
        pred = []
        y_true = []
        for imgs, labels in testloader:
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            with torch.no_grad():
                predicts = self.classify(imgs, task)
            pred.append(predicts)
            y_true.append(labels)
        return pred, y_true

    def _compute_loss(self, imgs, target):
        output = self.net(imgs)
        if config.ce:
            return F.cross_entropy(output, target)
        target = utils.to_one_hot(target, self.num_class).to(config.device)
        return F.binary_cross_entropy_with_logits(output, target,)

    def _pretrain(self):
        if config.dataset == 'iCIFAR100':
            from data import iCIFAR10
            pretrain_net = Base(
                iCIFAR10(train=True, increment=10), 
                iCIFAR10(train=False, increment=10), 
                pretrained=False
            )
            pretrain_net.name = "Pretrain_icifar100"
            pretrain_net.net.expand_fc(10)
            if not pretrain_net.resume(0, pretrain_net.name):
                pretrain_net.train(0)
            pretrain_net.eval(0)
            self.net.feature = pretrain_net.net.feature
            for param in self.net.feature.parameters():
                param.requires_grad = False
        # elif config.dataset == 'iImageNet_Subset':
        else:
            raise ValueError("Pretrained could not be used on dataset {}".format(config.dataset))

    def _construct_basic_net(self):
        '''
           may need to overwrite this func if you want to use a new NN. 
        '''
        basic_net = Basic_net()
        basic_net.fc_module = torch.nn.Linear
        basic_net.bias = True
        return basic_net
    
    


