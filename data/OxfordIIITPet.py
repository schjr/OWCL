# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : OxfordIIITPet.py
# DATE : 2022/8/28 13:01
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
import pickle as pkl
from config.config import oxfordiiitpet_root
from data.utils import DiscoverTargetTransform
from collections import Counter


# 37: 19 seen, 18 unseen
class OxfordIIITPet(data.Dataset):
    def __init__(self, root, indexs=None, split="trainval", transform=None, target_transform=None):
        super(OxfordIIITPet).__init__()
        self._anno_folder = os.path.join(root, "annotations")
        self._img_folder = os.path.join(root, "images")
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(self._anno_folder, f"{split}.txt")) as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                self.data.append(image_id + ".jpg")
                self.targets.append(int(label) - 1)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img_path = os.path.join(self._img_folder, img_path)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_oxfordiiitpet_datasets(transform_train, transform_val, num_labeled_classes,
                              num_unlabeled_classes, ratio, regenerate=False):
    train_dataset = OxfordIIITPet(root=oxfordiiitpet_root, transform=transform_train, split="trainval")
    val_dataset = OxfordIIITPet(root=oxfordiiitpet_root, transform=transform_val, split="trainval")
    test_dataset = OxfordIIITPet(root=oxfordiiitpet_root, transform=transform_val, split="test")

    dump_path = os.path.join("data/splits", f'oxfordiiitpet-labeled-{num_labeled_classes}'
                                            f'-unlabeled-{num_unlabeled_classes}-ratio-{int(ratio)}.pkl')

    targets = train_dataset.targets
    if regenerate or not os.path.exists(dump_path):
        class_idx = list(set(targets))
        shuffle_class_idx = np.arange(len(class_idx))
        np.random.shuffle(shuffle_class_idx)
        target_transform = {n: o for o, n in enumerate(shuffle_class_idx)}

        # select seen class
        seen_class = shuffle_class_idx[:num_labeled_classes]
        train_seen_idx = np.where(np.isin(targets, np.array(seen_class)))[0]
        targets_count = Counter(targets)
        number_per_class = np.zeros(len(class_idx))
        for i, seen in enumerate(train_seen_idx):
            number_per_class[targets[seen]] = int(targets_count[targets[seen]] * ratio / 100)

        train_indices_lab = []
        for idx in train_seen_idx:
            if number_per_class[targets[idx]] > 0:
                train_indices_lab.append(idx)
                number_per_class[targets[idx]] -= 1

        train_indices_unlab = np.array(list(set(range(targets.shape[0])) - set(train_indices_lab)))
        train_indices_lab = np.array(train_indices_lab)
        print(f"Number of labeled data: {len(train_indices_lab)}, number of unlabeled data: {len(train_indices_unlab)}")
        with open(dump_path, "wb") as f:
            pkl.dump({"label": train_indices_lab, "unlabel": train_indices_unlab,
                      "target_transform": target_transform}, f)
    else:
        with open(dump_path, "rb") as f:
            indices = pkl.load(f)

        train_indices_lab, train_indices_unlab, target_transform = \
            indices["label"], indices["unlabel"], indices["target_transform"]

    train_dataset.target_transform = DiscoverTargetTransform(target_transform)
    val_dataset.target_transform = DiscoverTargetTransform(target_transform)
    test_dataset.target_transform = DiscoverTargetTransform(target_transform)

    train_label_dataset = torch.utils.data.Subset(train_dataset, train_indices_lab)
    train_unlabel_dataset = torch.utils.data.Subset(train_dataset, train_indices_unlab)

    labeled_classes = np.array(list(target_transform.keys()))[:num_labeled_classes]

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