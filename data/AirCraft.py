# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : AirCraft.py
# DATE : 2022/8/28 13:01
import torch
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import os
from PIL import Image
import pickle as pkl
from config.config import aircraft_root
from data.utils import DiscoverTargetTransform
from collections import Counter


# 100: 50 seen, 50 unseen
class Aircraft(data.Dataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.
    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, class_type='variant', transform=None,
                 target_transform=None, download=False, indexs=None):
        super(Aircraft, self).__init__()
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, classes, class_to_idx) = self.find_classes()
        samples = self.make_dataset(image_ids, targets)

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        if indexs is not None:
            samples = []
            for index in indexs:
                samples.append(self.samples[index])
            self.samples = samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index

    def __len__(self):
        return len(self.samples)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        for i in range(len(image_ids)):
            item = (os.path.join(self.root, self.img_folder,
                                 '%s.jpg' % image_ids[i]), targets[i])
            images.append(item)
        return images


def get_aircraft_datasets(transform_train, transform_val, num_labeled_classes,
                              num_unlabeled_classes, ratio, regenerate=False):
    train_dataset = Aircraft(root=aircraft_root, transform=transform_train, train=True)
    val_dataset = Aircraft(root=aircraft_root, transform=transform_val, train=True)
    test_dataset = Aircraft(root=aircraft_root, transform=transform_val, train=False)

    dump_path = os.path.join("data/splits", f'aircraft-labeled-{num_labeled_classes}'
                                            f'-unlabeled-{num_unlabeled_classes}-ratio-{int(ratio)}.pkl')

    targets = [item[1] for item in train_dataset.samples]
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