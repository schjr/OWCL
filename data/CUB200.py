# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : CUB200.py
# DATE : 2022/8/28 13:01
import torch
from torch.utils.data import Dataset
import torchvision
from scipy import io
from config.config import cub_root
from utils.utils import *
import numpy as np
import os
from PIL import Image
import pandas as pd
import pickle as pkl


# 200 class, 100 seen class, unseen class
class CUB200(Dataset):
    base_folder = 'images'

    def __init__(self, root, train=True, transform=None, is_seen=True, num_labeled_classes=160,
                 num_unlabeled_classes=40, ratio=50, regenerate=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train
        self.ratio = ratio
        self.regenerate = regenerate
        self.dump_path = os.path.join("data/splits", f'CUB200-labeled-{num_labeled_classes}'
                                                     f'-unlabeled-{num_unlabeled_classes}-ratio-{int(ratio)}.pkl')
        self._load_metadata(is_seen=is_seen, class_num=num_labeled_classes)
        self.targets = self.data["target"] - 1

    def _load_metadata(self, is_seen, class_num):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        # select seen class or unseen class
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
            all_lens = self.data.shape[0]
            if is_seen:
                if os.path.exists(self.dump_path) and not self.regenerate:
                    with open(self.dump_path, "rb") as f:
                        idx = pkl.load(f)
                    labeled_idx, unlabeled_idx = idx["label"], idx["unlabel"]
                else:
                    print("Regenerate data !")
                    self.data = self.data[self.data.target <= class_num]
                    lens = self.data.shape[0]
                    labeled_idx = np.random.choice(lens, int(lens * self.ratio / 100), replace=False)
                    unlabeled_idx = np.array([x for x in range(all_lens) if x not in labeled_idx])
                    with open(self.dump_path, "wb") as f:
                        pkl.dump({"label": labeled_idx, "unlabel": unlabeled_idx}, f)

                self.data = self.data.iloc[labeled_idx]
            else:
                if os.path.exists(self.dump_path):
                    with open(self.dump_path, "rb") as f:
                        idx = pkl.load(f)
                    unlabeled_idx = idx["unlabel"]
                else:
                    raise FileNotFoundError("Unlabel file has not been generated!!!")
                self.data = self.data.iloc[unlabeled_idx]

        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, idx


def get_cub200_datasets(transform_train, transform_val, num_labeled_classes,
                        num_unlabeled_classes, ratio, regenerate=False):
    train_label_dataset = CUB200(cub_root, train=True, is_seen=True, num_labeled_classes=num_labeled_classes,
                                 num_unlabeled_classes=num_unlabeled_classes, transform=transform_train,
                                 ratio=ratio, regenerate=regenerate)
    train_unlabel_dataset = CUB200(cub_root, train=True, is_seen=False, num_labeled_classes=num_labeled_classes,
                                   num_unlabeled_classes=num_unlabeled_classes, transform=transform_train, ratio=ratio)
    val_unlab_dataset = CUB200(cub_root, train=True, is_seen=False, num_labeled_classes=num_labeled_classes,
                               num_unlabeled_classes=num_unlabeled_classes, transform=transform_val, ratio=ratio)
    test_dataset = CUB200(cub_root, train=False, num_labeled_classes=num_labeled_classes,
                          num_unlabeled_classes=num_unlabeled_classes, transform=transform_val, ratio=ratio)

    labeled_classes = np.arange(num_labeled_classes)
    # seen class
    val_indices_seen = np.where(np.isin(np.array(val_unlab_dataset.targets), labeled_classes))[0]
    val_seen_dataset = torch.utils.data.Subset(val_unlab_dataset, val_indices_seen)

    # testset
    test_indices_seen = np.where(np.isin(np.array(test_dataset.targets), labeled_classes))[0]
    test_seen_dataset = torch.utils.data.Subset(test_dataset, test_indices_seen)

    all_datasets = {"train_label_dataset": train_label_dataset, "train_unlabel_dataset": train_unlabel_dataset,
                    "val_dataset": val_unlab_dataset, "val_seen_dataset": val_seen_dataset,
                    "test_dataset": test_dataset, "test_seen_dataset": test_seen_dataset}
    return all_datasets


if __name__ == "__main__":
    pass
