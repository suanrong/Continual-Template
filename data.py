from torchvision.datasets import CIFAR100, ImageNet, MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset, WeightedRandomSampler, RandomSampler
import numpy as np
import copy
import config
config = config.config

class IncrementalDataset(object):
    def __init__(self, train, num_class, data, targets, transform, seed, increment, validation=False, init_task_size=0, subset=False):
        self.train = train
        self.class_order = list(range(num_class))
        if config.random_order:
            np.random.seed(seed)
            np.random.shuffle(self.class_order)
        self.data = data
        self.targets = np.array(list(map(lambda x: self.class_order.index(x), targets)))
        self.transform = transform
        self.indices = []
        self.where = []
        if subset:
            num_class = 100
            self.class_order = self.class_order[:100]
        for i in range(num_class):
            self.where.append(np.where(self.targets == i)[0])
        self.validation = validation
        if config.init_task_size > 0:
            self.class_split = [0] + np.arange(init_task_size, num_class + 1, increment).tolist()
        else:
            self.class_split = np.arange(0, num_class + 1, increment).tolist()
        if validation:
            self.validation_set = self._split_validation()
        self.where = np.array(self.where)

    def get_num_class(self, task):
        return self.class_split[task + 1]

    def get_validation_loader(self, task):
        if self.validation:
            return self.validation.get_loader(task)
        else:
            return None

    def get_loader(self, task, exemplar_data=None):
        classes = list(range(self.class_split[task], self.class_split[task + 1]))
        self._set_classes(classes)
        data = copy.copy(self)
        data.indices = copy.deepcopy(data.indices)
        if self.train:
            num_samples = config.total_complexity
            sampler = RandomSampler(data, num_samples=num_samples, replacement=True)
            if exemplar_data:
                config.logger.info("Exemplars size : {}".format(len(exemplar_data)))
                data = ConcatDataset([data, exemplar_data])
                l1 = len(self)
                l2 = len(exemplar_data)
                if self.train and config.batch_adjust >= 0:
                    if config.batch_adjust == 0:
                        config.logger.info("Adaptively Batch adjust")
                        batch_adjust = 1.0 / (task + 1)
                    else:
                        config.logger.info("Batch adjust new task proportation : {}".format(config.batch_adjust))
                        batch_adjust = config.batch_adjust
                    sampler = WeightedRandomSampler([batch_adjust / l1]*l1 + [(1.0-batch_adjust) / l2]*l2, num_samples=num_samples, replacement=True)
                else:
                    sampler = RandomSampler(data, num_samples=num_samples, replacement=True)
                    config.logger.info("No Batch adjust")
        else:
            sampler = RandomSampler(data, replacement=False)
        return DataLoader(dataset=data,
                            sampler=sampler,
                            # shuffle=True,
                            batch_size=config.batch_size,
                            num_workers=8)

    def _set_classes(self, classes):
        if len(classes) == 0:
            self.indices = []
            return
        self.indices = np.hstack(self.where[classes])
        np.random.shuffle(self.indices)

    def _split_validation(self):
        validation_set = copy.copy(self)
        validation_set.train = False
        validation_set.where = []
        for i in range(len(self.where)):
            np.random.shuffle(self.where[i])
            l = len(self.where[i])
            validation_set.where.append(self.where[i][:l // 10])
            self.where[i] = self.where[i][l // 10:]
        validation_set.where = np.array(validation_set.where)
        return validation_set

    def __getitem__(self, index):
        idx = self.indices[index]
        return self.transform(self.data[idx][0]), self.targets[idx]

    def __len__(self):
        return len(self.indices)

    def get_raw_images_from_class(self, class_id):
        ind = np.where(np.array(self.targets)==class_id)[0]
        return [self.data[i][0] for i in ind]


class iCIFAR100(IncrementalDataset):
    num_class = 100
    def __init__(self, train, increment, init_task_size=0, seed=0, validation=False):
        self.train = train
        self.dataset = CIFAR100(root='../dataset', train=train, download=True)
        self.data = self.dataset
        self.targets = self.dataset.targets
        self.seed = seed
        self.normalize = [transforms.ToTensor(),
                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
        if train:
            self.transform = transforms.Compose([transforms.RandomCrop((32,32),padding=4),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ColorJitter(brightness=0.24705882352941178),
                                                   *self.normalize])
        else:
            self.transform = transforms.Compose(self.normalize)
        self.flip_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                *self.normalize])
        super().__init__(self.train, self.num_class, self.data, self.targets, self.transform, seed, increment, validation, init_task_size)

class iCIFAR10(IncrementalDataset):
    num_class = 10
    def __init__(self, train, increment, init_task_size=0, seed=0, validation=False):
        self.train = train
        self.dataset = CIFAR10(root='../dataset', train=train, download=True)
        self.data = self.dataset
        self.targets = self.dataset.targets
        self.seed = seed
        self.normalize = [transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]
        if train:
            self.transform = transforms.Compose([transforms.RandomCrop((32,32),padding=4),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ColorJitter(brightness=0.24705882352941178),
                                                   *self.normalize])
        else:
            self.transform = transforms.Compose(self.normalize)
        self.flip_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                *self.normalize])
        super().__init__(self.train, self.num_class, self.data, self.targets, self.transform, seed, increment, validation, init_task_size)


class iImageNet_Subset(IncrementalDataset):
    num_class = 100
    def __init__(self, train, increment, init_task_size=0, seed=0, validation=False):
        self.train = train
        self.seed = seed
        if train:
            self.dataset = ImageNet(root='/DATA/DATANAS1/yaoxr/ILSVRC2012', split="train")
        else:
            self.dataset = ImageNet(root='/DATA/DATANAS1/yaoxr/ILSVRC2012', split="val")
        self.data = self.dataset
        self.targets = self.dataset.targets
        self.normalize = [transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        if train:
            self.transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   *self.normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   *self.normalize])
        self.flip_transform=transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.RandomHorizontalFlip(p=1.),
                                                *self.normalize])
        super().__init__(self.train, 1000, self.data, self.targets, self.transform, seed, increment, validation, init_task_size, subset=True)

class iMNIST(IncrementalDataset):
    num_class = 10
    def __init__(self, train, increment, init_task_size=0, seed=0, validation=False):
        self.train = train
        self.seed = seed
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        if train:
            self.dataset = MNIST(root='../dataset', train=True, transform=trsfm,download=True)
        else:
            self.dataset = MNIST(root='../dataset', train=False, transform=trsfm,download=True)
        self.data = self.dataset
        self.targets = self.dataset.targets
        self.transform = transforms.Compose([])
        self.flip_transform=transforms.Compose([])
        super().__init__(self.train, 10, self.data, self.targets, self.transform, seed, increment, validation, init_task_size)