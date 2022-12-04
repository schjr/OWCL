# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : orca_new.py
# DATE : 2022/8/29 22:29
# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : orca.py
# DATE : 2022/8/27 10:33

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import contextlib
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from data.get_dataset import get_discover_datasets
from model.resnet_s import resnet18
from model.resnet import resnet50
import os
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from utils.eval import split_cluster_acc_v1
from collections import defaultdict
import numpy as np
from losses.margin import *
from itertools import cycle
from utils.utils import model_statistics


def main(args):
    device = torch.device('cuda:0')

    # define dataset and dataloader
    dataset = get_discover_datasets(args.dataset, args)
    train_label_dataset, train_unlabel_dataset, test_dataset, val_dataset, test_seen_dataset = \
        dataset["train_label_dataset"], dataset["train_unlabel_dataset"], dataset["test_dataset"], \
        dataset["val_dataset"], dataset["test_seen_dataset"]

    labeled_len = len(train_label_dataset)
    unlabeled_len = len(train_unlabel_dataset)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    train_labeled_dataloader = torch.utils.data.DataLoader(
            train_label_dataset,
            batch_size=labeled_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    train_unlabeled_dataloader = torch.utils.data.DataLoader(
            train_unlabel_dataset,
            batch_size=args.batch_size - labeled_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # define model
    if args.arch == "resnet18":
        model = resnet18(num_classes=args.num_classes)
    elif args.arch == "resnet50":
        model = resnet50(num_classes=args.num_classes)
    else:
        assert NotImplementedError("Only support resnet18, resnet50!!!")

    # simclr pretrain
    state_dict = torch.load(args.pretrained)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()

    print("Fine tuning feature after layer4.")
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    model_statistics(model)

    # opt and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    # wandb
    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project, entity=args.entity, config=state, name=args.exp_name, dir="logs")

    # train
    for epoch in range(args.max_epochs):
        mean_uncert = test(args, model, test_dataloader, prefix="test", uncertainty=True)
        train(args, model, train_labeled_dataloader, train_unlabeled_dataloader, mean_uncert, optimizer, wandb)
        test_results = test(args, model, test_dataloader, prefix="test")
        train_results = test(args, model, val_dataloader, prefix="train")

        scheduler.step()

        # log
        lr = np.around(optimizer.param_groups[0]["lr"], 4)
        print("--Comment-{}--Epoch-[{}/{}]--LR-[{}]--All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".
              format(args.comment, epoch, args.max_epochs, lr, train_results["train/all/avg"]*100,
                     train_results["train/novel/avg"]*100, train_results["train/seen/avg"]*100))

        wandb.log(train_results)
        wandb.log(test_results)

    # save model
    torch.save(model.state_dict(), args.model_save_dir)


def swapped_prediction_pairwise(args, pos_pairs, probs, bce):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            pos_pair = pos_pairs[view]
            pos_prob = probs[other_view][pos_pair, :]
            pos_sim = torch.bmm(probs[view].view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
            ones = torch.ones_like(pos_sim)
            pos_sim = torch.clamp(pos_sim, 0, 1)

            loss += bce(pos_sim, ones)

    return loss / (args.num_large_crops * (args.num_crops - 1))


def train(args, model, train_labeled_dataloader, train_unlabeled_dataloader, m, optimizer, logger):
    model.train()
    bar = tqdm(train_labeled_dataloader)
    unlabel_loader_iter = cycle(train_unlabeled_dataloader)

    nlc = args.num_labeled_classes
    scaler = GradScaler()
    amp_cm = autocast() if args.amp else contextlib.nullcontext()
    bce = nn.BCELoss()
    ce = MarginLoss(m=-1 * m)

    for batch in bar:
        optimizer.zero_grad()

        # unpack multi-view images
        views_lab, labels_lab, _, = batch
        views_unlab, labels_unlab, _ = next(unlabel_loader_iter)

        images = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]

        images = [image.cuda() for image in images]
        labels_lab = labels_lab.cuda()

        with amp_cm:
            # forward
            outputs = []
            feats = []
            probs = []
            for view in images:
                o, f = model(view)
                outputs.append(o)
                feats.append(f)
                probs.append(F.softmax(o, dim=1))

            prob_v = torch.stack(probs, dim=0)
            prob_v = prob_v.unsqueeze(dim=1)

            pos_pairs_list = []
            labeled_len = len(labels_lab)
            for feat in feats[:args.num_large_crops]:
                feat_detach = feat.detach()
                feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
                cosine_dist = torch.mm(feat_norm, feat_norm.t())

                pos_pairs = []
                target_np = labels_lab.cpu().numpy()

                # label part
                for i in range(labeled_len):
                    target_i = target_np[i]
                    idxs = np.where(target_np == target_i)[0]
                    if len(idxs) == 1:
                        pos_pairs.append(idxs[0])
                    else:
                        selec_idx = np.random.choice(idxs, 1)
                        while selec_idx == i:
                            selec_idx = np.random.choice(idxs, 1)
                        pos_pairs.append(int(selec_idx))

                # unlabel part
                unlabel_cosine_dist = cosine_dist[labeled_len:, :]
                vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
                pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
                pos_pairs.extend(pos_idx)
                pos_pairs_list.append(pos_pairs)

            bce_loss = swapped_prediction_pairwise(args, pos_pairs_list, probs, bce)

            ce_loss = 0
            for output in outputs:
                ce_loss += ce(output[:labeled_len], labels_lab) / args.num_crops

            entropy_loss = 0
            for prob in probs:
                entropy_loss += entropy(torch.mean(prob, 0)) / args.num_crops

            loss = - entropy_loss + ce_loss + bce_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bar.set_postfix({"loss": "{:.2f}".format(loss.detach().cpu().numpy())})
        results = {
            "loss": loss.clone(),
            "entropy_loss": entropy_loss.clone(),
            "ce_loss": ce_loss.clone(),
            "bce_loss": bce_loss.clone(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(results)


@torch.no_grad()
def test(args, model, test_loader, prefix, uncertainty=False):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.cuda(), label.cuda()
            output, _ = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    mean_uncert = 1 - np.mean(confs)

    if uncertainty:
        return mean_uncert
    else:
        results = split_cluster_acc_v1(targets, preds, num_seen=args.num_labeled_classes)

        log = {}
        for key, value in results.items():
            log[prefix + "/" + key + "/" + "avg"] = round(value, 4)
        return log


if __name__ == "__main__":
    parser = ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
    parser.add_argument("--project", default="MI", type=str, help="wandb project")
    parser.add_argument("--entity", default="owssl", type=str, help="wandb entity")
    parser.add_argument("--offline", default=True, action="store_true", help="disable wandb")
    parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
    parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
    parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
    parser.add_argument("--resume", default=False, action="store_true", help="whether to use old model")
    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--ratio", type=float, default=50, help="the percentage of labeled data")
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")
    parser.add_argument('--amp', default=False, action="store_true", help='use mixed precision training or not')

    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    os.environ["WANDB_API_KEY"] = "8a2383b05d272ebfdf491e9e9a9f616a24323853"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.model_save_dir = os.path.join(args.checkpoint_dir, "discover",
                                       f"{args.dataset}-{args.arch}-labeled-{args.num_labeled_classes}-"
                                       f"unlabeled-{args.num_unlabeled_classes}-ratio-{int(args.ratio)}.pth")
    args.exp_name = "-".join(["discover", args.arch, args.dataset, args.comment,
                              str(args.num_labeled_classes), str(args.num_unlabeled_classes), str(int(args.ratio))])
    main(args)


