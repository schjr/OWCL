# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : cifar.py
# DATE : 2022/8/27 14:14
# TODO: cifar10 and cifar100 dataloader
# 1. Generate data split
# 2. Supervised Pretrain dataloader
# 3. Discovery dataloader
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np
import os
import pickle as pkl
from config.config import cifar_10_root, cifar_100_root

import ipdb

class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def get_cifar_datasets(transform_train, transform_val, num_labeled_classes,
                       num_unlabeled_classes, ratio, dataset="CIFAR100",
                       regenerate=False, transform_uncr=None):
    if dataset == "CIFAR10":
        train_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_train)
        val_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_val)
        test_dataset = CustomCIFAR10(cifar_10_root, train=False, transform=transform_val)
        uncr_dataset = CustomCIFAR10(cifar_10_root, train=True, transform=transform_uncr)


    elif dataset == "CIFAR100":
        # label的temp怎么办呢
        train_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_train)
        val_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_val)
        test_dataset = CustomCIFAR100(cifar_100_root, train=False, transform=transform_val)
        uncr_dataset = CustomCIFAR100(cifar_100_root, train=True, transform=transform_uncr)

    else:
        raise NotImplementedError("cifar10 or cifar100")

    # split dataset
    dump_path = os.path.join("data/splits", f'{dataset}-labeled-{num_labeled_classes}'
                                            f'-unlabeled-{num_unlabeled_classes}-ratio-{int(ratio)}.pkl')
    labeled_classes = range(num_labeled_classes)

    if regenerate:
        train_indices_lab = []
        for lc in labeled_classes:
            idx = np.nonzero(np.array(train_dataset.targets) == lc)[0]  # labeled里面每一类的位置
            idx = np.random.choice(idx, int(ratio * len(idx) / 100), False)
            train_indices_lab.extend(idx)

        train_indices_unlab = np.array(list(set(range(len(train_dataset))) - set(train_indices_lab)))
        train_indices_lab = np.array(train_indices_lab)
        with open(dump_path, "wb") as f:
            pkl.dump({"lab": train_indices_lab, "unlab": train_indices_unlab}, f)
    else:
        if not os.path.exists(dump_path):
            raise FileNotFoundError(f"Dump_path is not exists: {dump_path}")

        with open(dump_path, "rb") as f:
            indices = pkl.load(f)
        train_indices_lab, train_indices_unlab = indices["lab"], indices["unlab"]

    train_label_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab)
    train_unlabel_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)
    uncr_unlabel_dataset = torch.utils.data.Subset(uncr_dataset, train_indices_unlab)


    # seen class
    val_indices_seen = np.where(np.isin(np.array(val_dataset.targets), labeled_classes))[0]
    val_indices_seen = np.intersect1d(val_indices_seen, train_indices_unlab)
    val_seen_dataset = torch.utils.data.Subset(val_dataset, val_indices_seen)
    val_unlab_dataset = torch.utils.data.Subset(val_dataset, train_indices_unlab)

    # test set
    test_indices_seen = np.where(np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_seen_dataset = torch.utils.data.Subset(test_dataset, test_indices_seen)
    test_indices_unseen = np.where(~np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_unseen_dataset = torch.utils.data.Subset(test_dataset, test_indices_unseen)
    # ipdb.set_trace()
    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_dataset": val_unlab_dataset, "val_seen_dataset": val_seen_dataset,
                    "test_dataset": test_dataset, "test_seen_dataset": test_seen_dataset,
                    "test_unseen_dataset": test_unseen_dataset, "uncr_unlabel_dataset":uncr_unlabel_dataset}
    return all_datasets


if __name__ == "__main__":
    pass
