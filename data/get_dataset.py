# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : get_dataset.py
# DATE : 2022/8/27 17:14
from data.utils import DiscoverDataset
from data.augmentations import get_transforms
from data.cifar import get_cifar_datasets
# from data.herbarium_19 import get_herbarium_datasets
from data.StanfordCars import get_stanford_datasets
from data.Imagenet100 import get_imagenet100_datasets
from data.TinyImagenet import get_tinyimagenet_con_datasets, get_tinyimagenet_datasets
# from data.CUB200 import get_cub200_datasets
# from data.fgvc_aircraft import get_aircraft_datasets
#
# from data.cifar import subsample_classes as subsample_dataset_cifar
# from data.herbarium_19 import subsample_classes as subsample_dataset_herb
# from data.stanford_cars import subsample_classes as subsample_dataset_scars
# from data.imagenet import subsample_classes as subsample_dataset_imagenet
# from data.cub import subsample_classes as subsample_dataset_cub
# from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
import functools

# ====== #
from torch.utils.data import Subset, Dataset
import numpy as np
import torch
import ipdb
from data.cifar_con import get_cifar_continual_datasets
from model.MultiHeadResNet import MultiHeadResNetWithDirectFeats
import pickle as pkl
import os
import sys


get_dataset_funcs = {
    'CIFAR10': functools.partial(get_cifar_datasets, dataset="CIFAR10"),
    'CIFAR100': functools.partial(get_cifar_datasets, dataset="CIFAR100"),
    'CIFAR10Con': functools.partial(get_cifar_continual_datasets, dataset="CIFAR10"),
    'CIFAR100Con': functools.partial(get_cifar_continual_datasets, dataset="CIFAR100"),
    'ImageNet100': get_imagenet100_datasets,
    'TinyImageNet': get_tinyimagenet_datasets,
    'TinyImageNetCon': get_tinyimagenet_con_datasets,
    # 'herbarium_19': get_herbarium_datasets,
    # 'CUB200': get_cub200_datasets,
    # 'aircraft': get_aircraft_datasets,
    'StanfordCars': get_stanford_datasets
}


def get_discover_datasets(dataset_name, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform
    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )
    transform_uncr = get_transforms(
        "unsupervised",
        args.dataset,
        num_large_crops=10,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes,
                             args.ratio, regenerate=args.regenerate, transform_uncr=transform_uncr)

    # Train split (labelled and unlabelled classes) for training
    datasets["train_dataset"] = DiscoverDataset(datasets['train_label_dataset'], datasets['train_unlabel_dataset'])
    print("Lens of train dataset: {}, lens of val dataset: "
          "{}".format(len(datasets["train_label_dataset"]), len(datasets["val_dataset"])))
    return datasets


def get_supervised_datasets(dataset_name, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError
    # Get transform

    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes,
                             args.num_unlabeled_classes, args.ratio, regenerate=args.regenerate)

    # Train split (labelled and unlabelled classes) for training
    train_dataset = datasets['train_label_dataset']
    test_dataset = datasets['test_seen_dataset']
    print("Lens of train dataset: {}, lens of val dataset: "
          "{}".format(len(datasets["train_label_dataset"]), len(datasets["val_dataset"])))

    return train_dataset, test_dataset

def get_TRSSL_datasets(dataset_name, args, temp_uncr):
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform

    transform_train = get_transforms(
            "unsupervised",
            args.dataset,
            multicrop=args.multicrop,
            num_large_crops=args.num_large_crops,
            num_small_crops=args.num_small_crops,
    )
    transform_uncr = get_transforms(
        "unsupervised",
        args.dataset,
        num_large_crops=10,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes,
                             args.num_unlabeled_classes, args.ratio, args.imb_factor, regenerate=args.regenerate,
                             temperature=args.temperature, temp_uncr=temp_uncr, transform_uncr=transform_uncr)

    # Train split (labelled and unlabelled classes) for training
    datasets["train_dataset"] = DiscoverDataset(datasets['train_label_dataset'], datasets['train_unlabel_dataset'])

    return datasets




# ========== #
# Deprecated #
def get_simple_replay_datasets(dataset_name, args):

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform
    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )
    transform_uncr = get_transforms(
        "unsupervised",
        args.dataset,
        num_large_crops=10,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes,
                             args.ratio, regenerate=args.regenerate, transform_uncr=transform_uncr)

    # For each labelled class, we choose the first two elements for replay
    memory_replay_indices = []
    for i in range(50):
        memory_replay_indices.append(i * 500) #The labelled training set contains 25000 data with 50 classes. ratio is 100.
        memory_replay_indices.append(i * 500 + 1) #The 500 should be num_of_all_samples*(num_labelled/(num_labelled+num_unlabelled))*(ratio/100)*(1/num_labelled)
    train_labelled_dataset = datasets['train_label_dataset']
    memory_replay_dataset = Subset(dataset=train_labelled_dataset, indices=memory_replay_indices)

    # We have to add the labelled training set into the validation set, otherwise during the training, the seen result is unknown.
    # There is still doubt about what to be put into validation set.
    datasets["val_dataset"].indices = np.concatenate((datasets["val_dataset"].indices, np.array(memory_replay_indices)))

    datasets["train_dataset"] = DiscoverDataset(memory_replay_dataset, datasets['train_unlabel_dataset'])
    print("Lens of simple replay dataset: {}, lens of val dataset: "
          "{}".format(len(memory_replay_dataset), len(datasets["val_dataset"])))
    return datasets


# ========== #
# Deprecated #
def get_mean_var_datasets(dataset_name, args):
    """
    :return: train_dataset: MergedDataset which concatenates pseudo labelled and all unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform
    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )
    transform_uncr = get_transforms(
        "unsupervised",
        args.dataset,
        num_large_crops=10,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes,
                             args.ratio, regenerate=args.regenerate, transform_uncr=transform_uncr)

    # For each labelled class, we generate num_replicates pseudo data for replay
    num_labeled_classes = 50
    train_labelled_dataset = datasets['train_label_dataset']
    means = []
    stds = []
    for i in range(num_labeled_classes):
        # real_index stands for indexes of all data whose label is i
        real_index = np.nonzero(np.array(train_labelled_dataset.dataset.targets) == i)[0]
        # index is the real_index confined in train dataset
        index = np.intersect1d(real_index, train_labelled_dataset.indices)
        #tensorlist records all data in train dataset whose label is i
        tensorlist = []
        coset = Subset(train_labelled_dataset, index)
        for j in range(len(coset)):
            tensorlist.append(coset.dataset.dataset.data[j])
        tensorlist = torch.tensor(tensorlist, dtype=torch.float32)
        means.append(torch.mean(tensorlist))
        stds.append(torch.std(tensorlist))
    assert len(means) == num_labeled_classes
    pseudo_datas = []
    pseudo_targets = []
    num_replicates = 5
    for i in range(num_labeled_classes):
        for j in range(num_replicates):
            # each data has four views
            views_list = []
            for _ in range(2):
                views_list.append(torch.normal(means[i], stds[i], (3, 32, 32)))
            for _ in range(2):
                views_list.append(torch.normal(means[i], stds[i], (3, 18, 18)))
            pseudo_datas.append(views_list)
            pseudo_targets.append(i)
    
    my_pseudo_dataset = PseudoDataset(pseudo_datas, pseudo_targets)

    datasets["train_dataset"] = DiscoverDataset(my_pseudo_dataset, datasets['train_unlabel_dataset'])
    print("Lens of pseudo dataset: {}, lens of val dataset: "
          "{}".format(len(my_pseudo_dataset), len(datasets["val_dataset"])))
    return datasets


# ========== #

def get_raw_datasets(dataset_name, args):
    """
    :return: a raw dataset with no labelled data in training
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform
    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes,
                             val_ratio=args.val_ratio, regenerate=args.regenerate, choice_ratio=0,
                             label_val_ratio=50)
    datasets["train_dataset"] = datasets['train_unlabel_dataset']


    print("Lens of train dataset: {}, lens of val dataset: "
          "{}".format(len(datasets["train_dataset"]), len(datasets["val_dataset"])))
    return datasets


# ========== #

def get_replay_datasets(dataset_name, args):
    """
    :return: a replay dataset with some labelled data replayed
    """

    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform
    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes,
                             val_ratio=args.val_ratio, regenerate=args.regenerate, choice_ratio=args.choice_ratio,
                             label_val_ratio=args.label_val_ratio)

    val_unlabel_dataset = datasets["val_dataset"]
    val_label_dataset = datasets["val_seen_dataset"]
    datasets["train_dataset"] = DiscoverDataset(datasets['train_label_dataset'], datasets['train_unlabel_dataset'])

    datasets["val_dataset"].indices = np.concatenate((val_unlabel_dataset.indices, val_label_dataset.indices))

    print("Lens of train dataset: {}, lens of val dataset: "
          "{}".format(len(datasets["train_dataset"]), len(datasets["val_dataset"])))
    return datasets

# calculate mean and sig of the lab_dataloader's data
@torch.no_grad()
def calculate_mean_sig(args, model, lab_dataloader):
    model.eval()
    all_labels = None
    all_feats = None
    for batch in lab_dataloader:
        images_original, labels_original, _ = batch
        images = [i.cuda() for i in images_original]
        labels = labels_original.cuda()

        outputs = model(images, False)
        feats_original = outputs["feats"]
        feats = feats_original.cuda()

        if all_labels == None:
            all_labels = labels
            all_feats = feats[0]
        else:
            all_labels = torch.cat([all_labels, labels], dim=0)
            all_feats = torch.cat([all_feats, feats[0]], dim=0)

    class_mean = torch.zeros(args.num_labeled_classes, 512).cuda()
    class_sig = torch.zeros(args.num_labeled_classes, 512).cuda()
    for i in range(args.num_labeled_classes):
        this_feat = all_feats[all_labels==i]
        this_mean = this_feat.mean(dim=0)
        this_var = this_feat.var(dim=0)
        class_mean[i, :] = this_mean
        class_sig[i, :] = (this_var + 1e-5).sqrt()
    return class_mean, class_sig


# A template for pseudodataset
class PseudoDataset(Dataset):
    def __init__(self, PseudoDatas, PseudoTargets) -> None:
        self.datas = PseudoDatas
        self.targets = PseudoTargets
        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = self.datas[item], self.targets[item]
        uq_idx = self.uq_idxs[item]
        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def generate_pseudo_dataset(original_dataset, args, model, num_per_class):

    '''
    Return a dataset from original dataset, pseudo, Gaussian.
    The dataset contains feats and labels.
    '''

    pseudo_dataloader = torch.utils.data.DataLoader(
        original_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    mean, sig = calculate_mean_sig(args, model, pseudo_dataloader)

    feats = []
    labels = []

    for i in range(args.num_labeled_classes):
        dist = torch.distributions.Normal(mean[i], sig[i])
        this_feat = dist.sample((num_per_class,)).cuda()
        this_label = torch.ones(this_feat.size(0)).cuda() * i

        feats.append(this_feat)
        labels.append(this_label)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0).long()

    return PseudoDataset(feats, labels)

def get_pseudo_replay_datasets(dataset_name, args):
    """
    :return: a pseudo replay dataset with some pseudo data replayed
    """


    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError(f"dataset_name:{dataset_name}, not in {list(get_dataset_funcs.keys())}")
    # Get transform
    transform_train = get_transforms(
        "unsupervised",
        args.dataset,
        multicrop=args.multicrop,
        num_large_crops=args.num_large_crops,
        num_small_crops=args.num_small_crops,
    )
    transform_val = get_transforms("eval", args.dataset)

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(transform_train, transform_val, args.num_labeled_classes, args.num_unlabeled_classes,
                             val_ratio=args.val_ratio, regenerate=args.regenerate, choice_ratio=args.choice_ratio,
                             label_val_ratio=args.label_val_ratio)

    # val_unlabel_dataset = datasets["val_dataset"]
    # val_label_dataset = datasets["val_seen_dataset"]

    regenerate_dataset = False
    '''    
    Whenever we try to change the number of pseudodata:
    First set regenerate_dataset to be True and run
    After seeing the CUDA error,
    set regenerate_dataset to be False and we can run now
    '''

    dump_path = os.path.join("data/splits", f'pseudo-dataset.pkl')
    if regenerate_dataset:
        device = torch.device('cuda:0')
        model = MultiHeadResNetWithDirectFeats(
            arch=args.arch,
            low_res="CIFAR" in args.dataset or "tiny" in args.dataset,
            num_labeled=args.num_labeled_classes,
            num_unlabeled=args.num_unlabeled_classes,
            proj_dim=args.proj_dim,
            hidden_dim=args.hidden_dim,
            overcluster_factor=args.overcluster_factor,
            num_heads=args.num_heads,
            num_hidden_layers=args.num_hidden_layers,
        )
        state_dict = torch.load(args.pretrained, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        num_pseudo_per_class = 500
        pseudo_dataset = generate_pseudo_dataset(datasets['train_label_dataset'], args, model, num_pseudo_per_class)
        with open(dump_path, "wb") as f:
            torch.save({"feats": pseudo_dataset.datas, "labels": pseudo_dataset.targets}, f)
        print("Dataset generated! Please run with regenerate closed again!")
        sys.exit(0)
    else:
        if not os.path.exists(dump_path):
            raise FileNotFoundError(f"Dump_path does not exists: {dump_path}")
        with open(dump_path, "rb") as f:
            datas = torch.load(f, map_location='cpu')
            data, target = datas["feats"], datas["labels"]
            pseudo_dataset = PseudoDataset(data, target)

    datasets["train_dataset"] = DiscoverDataset(pseudo_dataset, datasets['train_unlabel_dataset'])

    # datasets["val_dataset"].indices = np.concatenate((val_unlabel_dataset.indices, val_label_dataset.indices))
    print("Lens of train dataset: {}, lens of val dataset: "
          "{}".format(len(datasets["train_dataset"]), len(datasets["val_dataset"])))
    return datasets


