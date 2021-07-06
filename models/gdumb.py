import torch
from torch.utils.data import DataLoader

import utils
from .replay import Replay
import config
config = config.config

class GDumb(Replay):
    '''
        GDumb: A Simple Approach that Questions Our Progress in Continual Learning
    '''
    # def get_train_loader(self, task):
    #     config.logger.info("Adding new class exemplars")
    #     for class_id in range(self.last_class, self.num_class):
    #         self.exemplar_handler.add_class(class_id)
    #     self.exemplar_handler.shuffle()
    #     self.exemplar_handler.reduce_exemplar_sets()
    #     dataloader = DataLoader(dataset=self.exemplar_handler.get_exemplar_dataset(),
    #                         shuffle=True,
    #                         batch_size=config.batch_size,
    #                         num_workers=4)
    #     config.logger.info("Data Size: {}".format(len(dataloader.dataset)))
    #     return dataloader, None

    # def after_train(self, task):
    #     self.class_mean_set = self.exemplar_handler.comput_exemplar_class_mean()
    #     if config.visual:
    #         self.view(task)
    #     if config.review:
    #         self.review()


