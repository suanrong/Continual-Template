from pdb import Pdb
import time
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import numpy as np
import pandas as pd

import config
from config import setup_writer
config.init()
config = config.config
import utils
import data
from exemplar import ExemplarHandler
import models

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

def trial(seed):
    ## Setup dataset, exemplarHandler, model
    data_class = getattr(data, config.dataset)
    exemplar_class = ExemplarHandler
    model_class = getattr(models, config.model)
    model = model_class(
            data_class(train=True, seed=seed, validation=False, increment=config.increment, init_task_size=config.init_task_size), 
            data_class(train=False, seed=seed, increment=config.increment, init_task_size=config.init_task_size), 
            config.pretrained,
            exemplar_class)
    config.logger.info("Model {}".format(config.model))
    config.logger.info("Dataset {}".format(model.train_dataset))
    config.logger.info("Class order {}".format(model.train_dataset.class_order))
    config.logger.info("Classify Mode {}".format(config.classify_mode))

    ## Determine the number of tasks
    if config.init_task_size:
        num_task = (model.train_dataset.num_class - config.init_task_size) // config.increment + 1
    else:
        num_task = model.train_dataset.num_class // config.increment
    
    ## Start the trial
    result = []
    accuracy_curve = []
    saved_batch_size = config.batch_size
    for task in range(num_task):
        model.train(task)
        _, all_info, _ = model.eval(task)
        config.logger.info("Task {} accuracy {}".format(task, str(all_info)))
        result.append(all_info)
        accuracy_curve.append(all_info['total'])
        config.writer.add_scalar("Average_Acuracy", all_info['total'], model.num_class)
        config.batch_size += config.bs_increment
    config.batch_size = saved_batch_size
    config.logger.info("Incremental Average Accuracy: {} ".format(np.mean(accuracy_curve)))
    average_accuracy = result[-1]['total']
    config.logger.info("Average Accuracy: {} ".format(average_accuracy))
    average_forgetting = utils.calc_forgetting(result, model.train_dataset.class_split)
    config.logger.info("Average Forgetting: {} ".format(average_forgetting))
    save_result(average_accuracy, average_forgetting, accuracy_curve, num_task, seed)
    return average_accuracy, average_forgetting, np.mean(accuracy_curve)

def main():
    print(config)
    # Run several experiments with different random seed
    accs = []
    forgettings = []
    iaas = []
    for seed in config.seeds:
        utils.setup_seed(seed)
        setup_writer(seed)
        start_time = time.time()
        acc, forgetting, iaa = trial(seed)
        accs.append(acc)
        forgettings.append(forgetting)
        iaas.append(iaa)
        config.logger.info("Training finished in {}".format(int(time.time() - start_time)))

    config.logger.info("Final incremental average accuracy {} +- {} \n {}".format(np.mean(iaas), np.std(iaas), iaas))
    config.logger.info("Final average accuracy {} +- {} \n {}".format(np.mean(accs), np.std(accs), accs))
    config.logger.info("Final average forgetting {} +- {} \n {}".format(np.mean(forgettings), np.std(forgettings), forgettings))
    config.logger.info("log is saved in {}".format(config.log_dir))

if __name__ == "__main__":
    main()

    

