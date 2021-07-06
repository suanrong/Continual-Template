import torch
import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    fhlr = logging.FileHandler(log_dir + "/log.txt") # 输出到文件的handler
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger


# tensorboard
    # 'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
    # 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'

def to_one_hot(target, num_class):
    '''
        target = [1,5,2]
        num_class = 6
        return [[0,1,0,0,0,0], [0,0,0,0,0,1], [0,0,1,0,0,0]]
    '''
    one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
    one_hot = one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def calc_accuracy(ypred, ytrue, class_split):
    ## calculate accuracy for previous task
    all_acc = {}
    all_acc["total"] = round((ypred == ytrue).sum().item() / len(ytrue), 3)
    for i in range(len(class_split) - 1):
        st_class = class_split[i]
        en_class = class_split[i+1]
        if st_class > torch.max(ytrue):
            break
        idxes = torch.where(
                torch.logical_and(ytrue >= st_class, ytrue < en_class)
        )[0]

        label = "{}-{}".format(
                str(st_class).rjust(2, "0"),
                str(en_class - 1).rjust(2, "0")
        )
        all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum().item() / len(idxes), 3)

    return all_acc

def calc_forgetting(all_acc, class_split):
    ## calculate average forgetting
    forgetting = 0.0
    for i in range(len(all_acc) - 1):
        max_acc = 0
        st_class = class_split[i]
        en_class = class_split[i+1]
        label = "{}-{}".format(
                str(st_class).rjust(2, "0"),
                str(en_class - 1).rjust(2, "0")
        )
        for j in range(i, len(all_acc)):
            if all_acc[j][label] > max_acc:
                max_acc = all_acc[j][label]
        forgetting += max_acc - all_acc[-1][label]
    return forgetting / (len(all_acc) - 1)

def occumpy_mem(cuda_device):
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device('cuda:'+cuda_device)
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total*0.90)
    block_mem = max_mem - used
    x = torch.FloatTensor(256,1024,block_mem).to(device)
    del x

def check_mem(cuda_device):
    import os
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def calc_metrics(y_true, y_pred, class_id):
    confused = {}
    y_true = y_true == class_id
    for i in range(max(y_pred) + 1):
        confused[i] = ((y_pred == i) * y_true).sum().item() 
    confused = sorted(confused.items(), key=lambda x:x[1], reverse=True)
    y_pred = y_pred == class_id
    pf = y_pred.sum().item()
    TP = (y_true * y_pred).sum().item()
    FP = ((~y_true) * y_pred).sum().item()
    FN = (y_true * (~y_pred)).sum().item()
    # ACC = (TP + TN) / (TP + FP + FN + TN)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN)
    if (TP == 0):
        F1 = 0.0
    else:
        F1 = round(2 * precision * recall / (precision + recall), 3)
    # return {"loss": round(loss, 3), "Prec":round(precision, 3), "Reca":round(recall, 3), "F1":F1, "CF":confused[:2], "Pred freq":pf}
    return {"Precision":round(precision, 3), "Recall":round(recall, 3)}

def get_free_gpu():
    import os
    import gpustat

    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    pairs = list(zip(ids, ratios))
    bestGPU = min(pairs, key=lambda x: x[1])[0]
    return bestGPU

def images_transform(images, transform):
    data = [transform(image) for image in images]
    data = torch.stack(data)
    return data

def flatten_params(m):
    total_params = []
    for param in m.parameters():
        total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    return total_params.cpu().detach().numpy()

def assign_weights(m, weights):
    import torch.nn as nn
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            # if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
            #     continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(torch.from_numpy(weights[index:index+param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m