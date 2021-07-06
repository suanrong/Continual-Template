import numpy as np
import torch
from torch.nn import functional as F

from .replay import Replay
import config
config = config.config

### Deprecated ###

class IL2M(Replay):
    '''
        IL2M: Class Incremental Learning With Dual Memory
    '''
    def __init__(self, train_dataset, test_dataset, pretrained, exemplar_handler):
        super().__init__(train_dataset, test_dataset, pretrained, exemplar_handler)
        self.init_score = []
        self.cur_score = []
        self.model_confidence = []
        if config.init_task_size == 0:
            self.model_confidence.append(1.0)

    def after_train(self, task):
        super().after_train(task)
        self.eval(task)
        if task == 0 and config.init_task_size:
            st_class = 0
            init_score = np.zeros(config.init_task_size)
            init_count = np.zeros(config.init_task_size)
        else:
            st_class = self.num_class - config.increment
            init_score = np.zeros(config.increment)
            init_count = np.zeros(config.increment)
        self.cur_score = np.zeros(self.num_class)
        cur_count = np.zeros(self.num_class)
        model_confidence = 0
        mc_count = 0
        loader = self.train_dataset.get_loader(task, self.exemplar_handler.get_exemplar_dataset())
        for images, target in loader:
            images, target = images.to(config.device), target.to(config.device)
            with torch.no_grad():
                outputs = self.net(images)
            for i, tt in enumerate(target):
                # tt = torch.argmax(outputs[i]).item()
                if tt >= st_class:
                    init_score[tt - st_class] += outputs[i][tt]
                    init_count[tt - st_class] += 1
                self.cur_score[tt] += outputs[i][tt]
                cur_count[tt] += 1
                model_confidence += max(outputs[i])
                mc_count += 1
        self.model_confidence.append((model_confidence / mc_count).item())
        for i in range(0, self.num_class - st_class):
            init_score[i] /= init_count[i]
        self.init_score.extend(init_score)
        for i in range(self.num_class):
            self.cur_score[i] /= cur_count[i]
        
        config.logger.info("cur_score {}".format(self.cur_score))
        config.logger.info("init_score {}".format(self.init_score))
        config.logger.info("model_confidence {}".format(self.model_confidence))
        
    def classify(self, imgs, task, mode="CNN"):
        if mode == "CNN":
            outputs = self.net(imgs)
            if self.num_class == config.init_task_size and config.init_task_size:
                st_class = self.num_class - config.init_task_size
            else:
                st_class = self.num_class - config.increment
            for i in range(len(outputs)):
                if torch.argmax(outputs[i]).item() >= st_class:
                    for j in range(st_class):
                        if j < config.init_task_size:
                            outputs[i][j] *= self.init_score[j] / self.cur_score[j] * self.model_confidence[-1] / self.model_confidence[0]
                        else:
                            outputs[i][j] *= self.init_score[j] / self.cur_score[j] * self.model_confidence[-1] / self.model_confidence[(j-config.init_task_size) // config.increment + 1]
        elif mode == "NME":
            features = F.normalize(self.net.extract_feature(imgs).detach())
            outputs = torch.matmul(features, torch.transpose(self.class_mean_set, 0, 1))
        else:
            raise ValueError("unknown mode for classify", mode)
        if config.task_incre:
            st = self.train_dataset.class_split[task]
            en = self.train_dataset.class_split[task+1]
            return torch.argmax(outputs[:,st:en], 1) + self.train_dataset.class_split[task]
        return torch.argmax(outputs, dim=1)


