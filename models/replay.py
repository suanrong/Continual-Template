import torch
from torch.nn import functional as F

from .base import Base
import utils
import config
config = config.config


class Replay(Base):
    """
        This is the base class for method using exemplars
    """
    def __init__(self, train_dataset, test_dataset, pretrained, exemplar_handler, *args):
        super().__init__(train_dataset, test_dataset, pretrained)
        self.exemplar_handler = exemplar_handler(self, self.train_dataset)
        self.class_mean_set = None
        self.last_class = 0

    def expand_net(self, num_class):
        if num_class > self.num_class:
            self.last_class = self.num_class
            self.num_class = num_class
            self.net.expand_fc(self.num_class)
        self.net.to(config.device)

    def _get_train_loader(self, task):
        return self.train_dataset.get_loader(task, self.exemplar_handler.get_exemplar_dataset()), self.train_dataset.get_validation_loader(task)

    def after_train(self, task):
        super().after_train(task)
        config.logger.info("Adding new class exemplars")
        for class_id in range(self.last_class, self.num_class):
            self.exemplar_handler.add_class(class_id)
        self.exemplar_handler.shuffle()
        self.exemplar_handler.reduce_exemplar_sets()
        self.class_mean_set = self.exemplar_handler.comput_exemplar_class_mean()
        if config.finetune_on_exemplars:
            _, all_info, _ = self.eval(task)
            config.logger.info("Task {} betore finetune accuracy {}".format(task, str(all_info)))
            config.logger.info("-------------------- Task {} --------------------".format(task))
            self._setup_optimizer(scheduler="MultiStepLR")
            from torch.utils.data import DataLoader, RandomSampler
            D = self.exemplar_handler.get_exemplar_dataset()
            sampler = RandomSampler(D, num_samples=config.total_complexity, replacement=True)
            train_loader = DataLoader(dataset=D,
                                sampler=sampler,
                                batch_size=config.batch_size,
                                num_workers=8)
            self._train(task, train_loader)

        if config.visual:
            self.view(task)
        if config.review:
            self.review()

    def classify(self, imgs, task, mode=None):
        self.net.eval()
        if not mode:
            mode = config.classify_mode
        if mode == "CNN":
            outputs = self.net(imgs)
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

    def view(self, task):
        self.net.eval()
        f = []
        la = []
        imgs = []
        for class_id in range(10):
            images = self.train_dataset.get_raw_images_from_class(class_id)[:100]
            with torch.no_grad():
                feature = self.net.extract_feature(utils.images_transform(images, self.test_dataset.transform).to(config.device))
            f.append(feature)
            la.extend(len(images) * [class_id+1])
            imgs.append(images)
            # datas = self.exemplar_handler.utils.images_transform(images, self.transform).to(config.device)
            # outputs = self.NME_classify(datas).cpu()
            # validate_imgs = self.exemplar_handler.get_validation_set(class_id)
            # validate_feature = self.net.extract_feature(utils.images_transform(validate_imgs, self.test_dataset.transform).to(config.device))
            # # datas = self.exemplar_handler.utils.images_transform(validate_imgs, self.transform).to(config.device)
            # # validate_outputs = torch.ones(len(validate_imgs)).long() * -1
            # f.append(validate_feature)
            # la.extend(len(validate_imgs) * [-(class_id+1)])
            # imgs.append(validate_imgs)
        config.writer.add_embedding(torch.cat(f), metadata=la, global_step=task, tag="Class 0-9 Feature space at task {}".format(task))

    def review(self):
        config.logger.info("Reviewing")
        config.logger.info("Review strategy: {}".format(config.review_strategy))
        self.net.eval()
        pred = []
        y_true = []
        wrongs = []
        recall = {}
        review_class = []
        for class_id in range(self.num_class):
            imgs = self.exemplar_handler.get_validation_set(class_id)
            datas = utils.images_transform(imgs, self.test_dataset.transform).to(config.device)
            outputs = self.classify(datas, config.classify_mode)
            wrong = []
            for i, output in enumerate(outputs):
                if output != class_id:
                    wrong.append(i)
            wrong.sort()
            wrongs.append(wrong)
            recall[class_id] = 1 - len(wrong) / len(imgs)
            pred.append(outputs)
            y_true.append(torch.ones(len(outputs)).int() * class_id)
        pred = torch.cat(pred).to(config.device)
        y_true = torch.cat(y_true).to(config.device)

        result_info = {}
        for class_id in range(self.num_class):
            result_info[class_id] = utils.calc_metrics(y_true, pred, class_id)

        if config.access_limit >= 0:
            for class_id, v in sorted(result_info.items(), key=lambda x: x[1]["Recall"])[:config.access_limit]:
                self.exemplar_handler.refresh(class_id, wrongs[class_id])
                review_class.append(class_id)
        else:
            for class_id, v in sorted(result_info.items(), key=lambda x: x[1]["Recall"]):
                if v["Recall"] < config.threshold:
                    self.exemplar_handler.refresh(class_id, wrongs[class_id])
                    review_class.append(class_id)

        with open(config.log_dir + "/recall_class.txt","a+") as fout:
            for class_id in range(self.train_dataset.num_class):
                if class_id in recall:
                    print(recall[class_id], end=',', file=fout)
                else:
                    print("0.0", end=',', file=fout)
            print("", file=fout)
            for c in review_class:
                print(c, end=',', file=fout)
            print("", file=fout)

        self.class_mean_set = self.exemplar_handler.comput_exemplar_class_mean()

