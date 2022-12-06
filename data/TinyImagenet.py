# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : TinyImagenet.py
# DATE : 2022/8/28 13:00
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import os
import pickle as pkl
from config.config import tinyimagenet_root


# 200: 100 seen, 100 unseen
class TinyImagenet(datasets.ImageFolder):
    def __init__(self, root, indexs=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets = list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)

        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_tinyimagenet_datasets(transform_train, transform_val, num_labeled_classes,
                              num_unlabeled_classes, ratio, regenerate=False):
    train_data_dir = os.path.join(tinyimagenet_root, "train")
    val_data_dir = os.path.join(tinyimagenet_root, "val")
    number_per_class = 500
    train_dataset = TinyImagenet(root=train_data_dir, transform=transform_train)
    val_dataset = TinyImagenet(root=train_data_dir, transform=transform_val)
    test_dataset = TinyImagenet(root=val_data_dir, transform=transform_val)

    dump_path = os.path.join("data/splits", f'tinyimagenet-labeled-{num_labeled_classes}'
                                            f'-unlabeled-{num_unlabeled_classes}-ratio-{int(ratio)}.pkl')
    labeled_classes = range(num_labeled_classes)

    targets = train_dataset.targets

    if regenerate or not os.path.exists(dump_path):
        # 100 class as seen
        train_seen_idx = np.where(np.isin(targets, np.arange(num_labeled_classes)))[0]
        np.random.shuffle(train_seen_idx)
        number_per_class = np.ones(num_labeled_classes) * ratio * number_per_class / 100
        train_indices_lab = []
        for idx in train_seen_idx:
            if number_per_class[targets[idx]] > 0:
                train_indices_lab.append(idx)
                number_per_class[targets[idx]] -= 1

        train_indices_unlab = np.array(list(set(range(len(train_dataset))) - set(train_indices_lab)))
        train_indices_lab = np.array(train_indices_lab)
        print(
            f"Number of labeled data: {len(train_indices_lab)}, number of unlabeled data: {len(train_indices_unlab)}")
        with open(dump_path, "wb") as f:
            pkl.dump({"label": train_indices_lab, "unlabel": train_indices_unlab}, f)
    else:
        with open(dump_path, "rb") as f:
            indices = pkl.load(f)
        train_indices_lab, train_indices_unlab = indices["lab"], indices["unlab"]

    train_label_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab)
    train_unlabel_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)

    # seen class
    val_indices_seen = np.where(np.isin(np.array(val_dataset.targets), labeled_classes))[0]
    val_indices_seen = np.intersect1d(val_indices_seen, train_indices_unlab)
    val_seen_dataset = torch.utils.data.Subset(val_dataset, val_indices_seen)
    val_unlab_dataset = torch.utils.data.Subset(val_dataset, train_indices_unlab)

    # testset
    test_indices_seen = np.where(np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_seen_dataset = torch.utils.data.Subset(test_dataset, test_indices_seen)

    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_dataset": val_unlab_dataset, "val_seen_dataset": val_seen_dataset,
                    "test_dataset": test_dataset, "test_seen_dataset": test_seen_dataset}
    return all_datasets
