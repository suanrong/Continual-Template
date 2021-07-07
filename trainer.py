import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils
import data

from exemplar import ExemplarHandler
import models

import config
config = config.config

def save_result(average_accuracy, average_forgetting, accuracy_curve, num_task, seed):
    df = pd.DataFrame(
        {
            "Dataset":config.dataset,
            "Model":config.model,
            "Incre":config.increment,
            "Seed":seed,
            "Average_Acuracy":average_accuracy,
            "Average_Forgetting":average_forgetting,
            "Accuracy_curve":accuracy_curve,
            "Task":np.arange(num_task),
            "Classify":config.classify_mode,
            "Review":config.review,
            "Memory":config.memory_size
        }
    )
    df.to_csv(config.log_dir + "/"  + str(seed) + '.csv', index=False)

class ContinualTrainer(object):
    def __init__(self, seed):
        '''
            need to define:
                num_task
        '''
        self.seed = seed
        ## Setup dataset, exemplarHandler, model
        data_class = getattr(data, config.dataset)
        self.train_dataset = data_class(train=True, seed=seed, validation=False, increment=config.increment, init_task_size=config.init_task_size)
        self.test_dataset = data_class(train=False, seed=seed, increment=config.increment, init_task_size=config.init_task_size)
        exemplar_class = ExemplarHandler
        self.model = getattr(models, config.model)(self.train_dataset, self.test_dataset, config.pretrained, exemplar_class)
        config.logger.info("Model {}".format(config.model))
        config.logger.info("Dataset {}".format(self.train_dataset))
        config.logger.info("Class order {}".format(self.train_dataset.class_order))
        config.logger.info("Classify Mode {}".format(config.classify_mode))
        ## Determine the number of tasks
        if config.init_task_size:
            self.num_task = (self.train_dataset.num_class - config.init_task_size) // config.increment + 1
        else:
            self.num_task = self.train_dataset.num_class // config.increment

    def continual_train(self):
        result = []
        accuracy_curve = []
        for task in range(self.num_task):
            self.train_one_task(self.model, task)
            _, all_info, _ = self.eval_after_task(self.model, task)
            config.logger.info("Task {} accuracy {}".format(task, str(all_info)))
            result.append(all_info)
            accuracy_curve.append(all_info['total'])
            config.writer.add_scalar("Average_Acuracy", all_info['total'], self.model.num_class)
        
        config.logger.info("Incremental Average Accuracy: {} ".format(np.mean(accuracy_curve)))
        average_accuracy = result[-1]['total']
        config.logger.info("Average Accuracy: {} ".format(average_accuracy))
        average_forgetting = utils.calc_forgetting(result, self.model.train_dataset.class_split)
        config.logger.info("Average Forgetting: {} ".format(average_forgetting))
        save_result(average_accuracy, average_forgetting, accuracy_curve, self.num_task, self.seed)
        return average_accuracy, average_forgetting, np.mean(accuracy_curve)

    def train_one_task(self, model, task):
        config.logger.info("==================== Task {} ====================".format(task))

        model.expand_net(self.train_dataset.get_num_class(task))
        model.setup_optimizer()
        # if not model.resume(task):
        train_loader, valid_loader = self.model._get_train_loader(task)
        model.train(task, train_loader, valid_loader)
        model.save_model(task)
        model.after_train(task)

    def eval_after_task(self, model, current_task):
        preds = []
        y_trues = []
        all_acc = {}
        for task in range(current_task + 1):
            test_loader = self.test_dataset.get_loader(task)
            pred, y_true = model.test(test_loader, task)
            all_acc[task] = round(100.0 * (torch.cat(y_true) == torch.cat(pred)).sum().item() / len(pred), 3)
            preds.extend(pred)
            y_trues.extend(y_true)
        preds = torch.cat(preds); y_trues = torch.cat(y_trues)
        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(y_trues.cpu(), preds.cpu(), normalize='true')
        fig = plt.figure()
        plt.imshow(conf_mat)
        accuracy = round(100.0 * (preds == y_trues).sum().item() / len(preds), 3)
        all_acc['total'] = accuracy
        return accuracy, all_acc, fig


