import numpy as np
from torch.nn import functional as F
import torch
from torch.utils.data import Dataset
import copy

import config
config = config.config
from gng import GNG
import utils

class ExemplarDataset(Dataset):
    def __init__(self, datas, labels, transform):
        self.datas = datas
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.datas[idx]), self.labels[idx]

    def __len__(self):
        return len(self.datas)

class ExemplarHandler(object):
    '''
        The class that handles the exemplars.
        These exemplars are used to avoid forgetting & select classes for recalling. 
    '''
    def __init__(self, model, dataset):
        self.exemplars = []
        self.class_mean_set = []
        self.model = model
        self.transform = model.test_dataset.transform
        self.flip_transform = dataset.flip_transform
        self.train_transform = dataset.transform
        self.external_dataset = dataset
        if config.construct_strategy == 'herding':
            self._construct = self._herding
        elif config.construct_strategy == 'random_pick':
            self._construct = self._random_pick
        elif config.construct_strategy == 'kmeans':
            self._construct = self._kmeans
        else:
            raise NotImplementedError

        self.validation_sets = []
        if config.review_strategy == 'herding':
            self._review = self._herding
        elif config.review_strategy == 'random_pick':
            self._review = self._random_pick
        elif config.review_strategy == 'kmeans':
            self._review = self._kmeans
        else:
            raise NotImplementedError

        self._memory_per_class = config.memory_size

    def get_exemplar_dataset(self):
        datas, labels = [], []
        for class_id in range(len(self.exemplars)):
            exemplars = self._get_exemplars(class_id)
            datas.extend(exemplars)
            labels.extend(np.full(len(exemplars), class_id))
        if len(datas) == 0:
            exemplars_dataset =  ExemplarDataset([], [], self.train_transform)
        else:
            exemplars_dataset =  ExemplarDataset(datas, labels, self.train_transform)
        return exemplars_dataset

    def reduce_exemplar_sets(self):
        m = self._memory_per_class
        for i in range(len(self.exemplars)):
            self.exemplars[i] = self.exemplars[i][:m]
        
    def add_class(self, class_id):
        images = self.external_dataset.get_raw_images_from_class(class_id)
        if len(self.exemplars) <= class_id and config.fix_memory_size == True:
            self._memory_per_class = int(config.memory_size / (len(self.exemplars) + 1))
        exemplar = self._construct(images, self._memory_per_class)
        if len(self.exemplars) > class_id:
            self.exemplars[class_id] = exemplar
        else:
            self.exemplars.append(exemplar)
        if config.fix_memory_size == True:
            self.reduce_exemplar_sets()
        if config.review:
            validation_set = []
            self.validation_sets.append(validation_set)
            for _ in range(config.validation_size):
                index = np.random.randint(0,len(images))
                validation_set.append(images[index])
            if len(self.validation_sets) > class_id:
                self.exemplars[class_id] = exemplar
            else:
                self.exemplars.append(exemplar)

    def comput_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplars)):
            exemplar = self._get_exemplars(index)
            class_mean, _ = self._compute_mean_and_features(exemplar, self.transform)
            class_mean_flip, _ = self._compute_mean_and_features(exemplar, self.flip_transform)
            class_mean = F.normalize(class_mean, dim=0)
            class_mean_flip  = F.normalize(class_mean_flip, dim=0)
            class_mean = (class_mean + class_mean_flip) / 2
            self.class_mean_set.append(class_mean)
        return torch.stack(self.class_mean_set)

    def shuffle(self):
        for class_id in range(len(self)):
            np.random.shuffle(self.exemplars[class_id])

    def _compute_mean_and_features(self, images, transform):
        x = utils.images_transform(images, transform).to(config.device)
        feature_extractor_output = []
        with torch.no_grad():
            for i in range(len(x) // 128 + 1): 
                en = min(len(x), (i+1) * 128)
                feature_extractor_output.append(self.model.net.extract_feature(x[i*128:en]).detach())
                if en == len(x):
                    break
        feature_extractor_output = F.normalize(torch.cat(feature_extractor_output))
        class_mean = torch.mean(feature_extractor_output, 0)
        return class_mean, feature_extractor_output

    def __len__(self):
        return len(self.exemplars)

    def _get_exemplars(self, class_id):
        return self.exemplars[class_id]
        
    def _herding(self, images, size):
        result = []
        used_index = {}
        class_mean, feature_extractor_output = self._compute_mean_and_features(images, self.transform)
        w_t = class_mean
        D = torch.transpose(feature_extractor_output, 0, 1)
        step = 0
        while len(result) != size and step < 1.1 * size:
            tmp_t = torch.matmul(w_t, D)
            index = torch.argmax(tmp_t)
            w_t = w_t + class_mean - D[:, index]
            step += 1
            if index not in used_index:  
                result.append(images[index])
                used_index[index] = True
        return result

    def _random_pick(self, images, size):
        result = []
        for _ in range(size):
            index = np.random.randint(0, len(images))
            result.append(images[index])
        return result

    def _kmeans(self, images, size):
        import sklearn.cluster as cluster
        _, features = self._compute_mean_and_features(images, self.transform)
        kmeans = cluster.KMeans(n_clusters=size, random_state=0).fit(features.cpu())
        centoids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(config.device)
        distance = torch.matmul(centoids, features.transpose(0, 1))
        index = torch.argmax(distance, 1)
        result = []
        for ind in index:
            result.append(images[ind])
        return result
    
    def _model_pick(self, images, size):
        cnt = 0
        datas = utils.images_transform(images, self.model.transform).to(config.device)
        outputs = self.model.classify(datas)
        # for i, output in enumerate(outputs):
        #     if output != class_id:
        #         exemplar.append(images[i])
        #         cnt += 1
        #         if cnt == m:
        #             break

    def NG(self, class_id):
        # if config.NG_validate:
        #     gng = GNG(config.validation_size, len(self.exemplars) - 1)
        #     _, feature_extractor_output = self._compute_mean_and_features(images, self.transform)
        #     gng.fit_data(feature_extractor_output, images)
        #     self.validation_sets.append(gng.get_exemplars())
        # else:
        
        # # random strategy
        pass

    def get_validation_set(self, class_id = None):
        datas = []
        labels = []
        if class_id == None:
            for class_id in range(len(self.validation_sets)):
                validation_set = self.validation_sets[class_id]
                datas.extend(validation_set)
                labels.extend(np.full(len(validation_set), class_id))
            return ExemplarDataset(datas, labels, self.train_transform)
        else:
            datas = self.validation_sets[class_id]
            labels = np.full(len(datas), class_id)
            return datas


    def _check_validate(self, class_id, images):
        _, feature = self._compute_mean_and_features(images, self.transform)
        _, validate_feature = self._compute_mean_and_features(self.validation_sets[class_id], self.transform)
        error = 0.0
        feature = F.normalize(feature)
        validate_feature = F.normalize(validate_feature)
        for f in feature:
            dist = torch.matmul(f, validate_feature.transpose(0,1))
            error += dist.max().item()
        config.logger.info("Error for validation set class {} : {}".format(class_id, round(error, 3)))

    def refresh(self, class_id, wrong_id):
        images = self.external_dataset.get_raw_images_from_class(class_id)
        m = self._memory_per_class
        self.exemplars[class_id] = self._review(images, m)
        # dist strategy
        # if config.review_strategy == "dist":
        #     if len(wrong_id) == 0:
        #         return
        #     _, feature = self._compute_mean_and_features(imgs, self.transform)
        #     vali_imgs = self.validation_sets[class_id]
        #     _, vali_feature = self._compute_mean_and_features(vali_imgs, self.transform)
        #     vali_feature = F.normalize(vali_feature[wrong_id])
        #     dist = torch.matmul(feature, vali_feature.transpose(0,1))
        #     dist = dist.max(1)[0]
        #     ind = dist.argsort(descending=True)[:m]
        #     for i in ind:
        #         exemplar.append(imgs[i.item()])
        config.logger.info("Recall class {} size {}".format(class_id, len(self.exemplars[class_id])))



# class CH(RWR):
#     def refresh(self, class_id, wrong_id):
#         imgs = self.external_dataset.get_raw_images_from_class(class_id)
#         m = len(self.exemplars[class_id])
#         self.exemplars[class_id] = self.exemplars[class_id][:m // 2]
#         cnt = m // 2
#         datas = utils.images_transform(imgs, self.model.transform).to(config.device)
#         outputs = self.model.net(datas)
#         for i in sorted(zip(np.arange(len(outputs)), outputs), key = lambda x: x[1][class_id]):
#             self.exemplars[class_id].append(imgs[i[0]])
#             cnt += 1
#             if cnt == m:
#                 break
#         np.random.shuffle(self.exemplars[class_id])
#         config.logger.info("Recall class {} size {}".format(class_id, len(self.exemplars[class_id])))