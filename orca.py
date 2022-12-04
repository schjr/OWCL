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
    train_label_dataset, train_unlabel_dataset, test_dataset, val_dataset = \
        dataset["train_label_dataset"], dataset["train_unlabel_dataset"], dataset["test_dataset"], dataset["val_dataset"]

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
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('/model_path/')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, args.max_epochs):
        train_results, mean_uncert = test(args, model, val_dataloader, prefix="train")
        if args.margin_type == "dnm":
            # mean_uncert = min(mean_uncert, 0.5)
            mean_uncert = 0.5 * (1 - epoch / (args.max_epochs - 1))
        elif args.margin_type == "fnm":
            mean_uncert = 0.5
        elif args.margin_type == "znm":
            mean_uncert = 0
        else:
            assert NotImplementedError("dnm, fnm, znm, but input is {}".format(args.margin_type))

        train(args, model, train_labeled_dataloader, train_unlabeled_dataloader, mean_uncert, optimizer, wandb)
        test_results, _ = test(args, model, test_dataloader, prefix="test")

        scheduler.step()

        # log
        lr = np.around(optimizer.param_groups[0]["lr"], 4)
        print("--Comment-{}--Epoch-[{}/{}]--LR-[{}]--All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]--Uncert-[{:.2f}]".
              format(args.comment, epoch, args.max_epochs, lr, train_results["train/all/avg"]*100,
                     train_results["train/novel/avg"]*100, train_results["train/seen/avg"]*100, mean_uncert))

        wandb.log(train_results)
        wandb.log(test_results)
        wandb.log({"mean_uncert": mean_uncert})
        # save model
        if args.save_model:
            model_to_save = model.module if hasattr(model, "module") else model
            state_save = {
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = args.exp_name + '_' + str(epoch) + f'checkpoint.pth.tar'
            filepath = os.path.join(args.model_save_dir, filename)
            torch.save(state_save, filepath)



def train(args, model, train_labeled_dataloader, train_unlabeled_dataloader, m, optimizer, logger):
    model.train()
    bar = tqdm(train_labeled_dataloader)
    unlabel_loader_iter = cycle(train_unlabeled_dataloader)

    scaler = GradScaler()
    amp_cm = autocast() if args.amp else contextlib.nullcontext()
    bce = nn.BCELoss()
    # m = min(m, 0.5)
    ce = MarginLoss(m=-1 * m)
    log_bce = 0
    log_ce = 0
    log_entropy = 0
    for batch in bar:

        # unpack multi-view images
        views_lab, target, _, = batch
        views_unlab, unlab_target, _ = next(unlabel_loader_iter)

        x = torch.cat([views_lab[0], views_unlab[0]], 0)
        x2 = torch.cat([views_lab[1], views_unlab[1]], 0)
        x, x2, target = x.cuda(), x2.cuda(), target.cuda()
        # images = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
        #
        # images = [image.cuda() for image in images]
        # target = target.cuda()

        with amp_cm:
            # forward
            output, feat = model(x)
            output2, feat2 = model(x2)
            prob = F.softmax(output / args.temperature, dim=1)
            prob2 = F.softmax(output2 / args.temperature, dim=1)

            # first view generate pair wise pseudo label for the second view
            feat_detach = feat.detach()
            feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
            cosine_dist = torch.mm(feat_norm, feat_norm.t())
            labeled_len = len(target)

            pos_pairs = []
            target_np = target.cpu().numpy()

            # label part
            for i in range(labeled_len):
                target_i = target_np[i]
                idxs = np.where(target_np == target_i)[0]
                if len(idxs) == 1:
                    pos_pairs.append(idxs[0])
                else:
                    select_idx = np.random.choice(idxs, 1)
                    while select_idx == i:
                        select_idx = np.random.choice(idxs, 1)
                    pos_pairs.append(int(select_idx))

            # unlabel part
            unlabel_cosine_dist = cosine_dist[labeled_len:, :]
            vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)

            pos_prob = prob2[pos_pairs, :]
            pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
            ones = torch.ones_like(pos_sim)

            bce_loss = bce(pos_sim, ones)
            ce_loss = ce(output[:labeled_len], target)
            entropy_loss = entropy(torch.mean(prob, 0))

            log_entropy += entropy_loss
            log_ce += ce_loss
            log_bce += bce_loss

            loss = - entropy_loss + ce_loss + bce_loss

        optimizer.zero_grad()
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
def test(args, model, test_loader, prefix):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.cuda(), label.cuda()
            output, _ = model(x)
            prob = F.softmax(output / args.temperature, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    mean_uncert = 1 - np.mean(confs)
    results = split_cluster_acc_v1(targets, preds, num_seen=args.num_labeled_classes)

    log = {}
    for key, value in results.items():
        log[prefix + "/" + key + "/" + "avg"] = round(value, 4)

    return log, mean_uncert


if __name__ == "__main__":
    parser = ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers")
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
    parser.add_argument("--project", default="MI", type=str, help="wandb project")
    parser.add_argument("--entity", default="owssl", type=str, help="wandb entity")
    parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
    parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
    parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
    parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--ratio", type=float, default=50, help="the percentage of labeled data")
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")
    parser.add_argument('--amp', default=False, action="store_true", help='use mixed precision training or not')
    parser.add_argument("--resume", default=False, action="store_true", help="whether to use old model")
    parser.add_argument("--save-model", default=False, action="store_true", help="whether to save model")

    parser.add_argument('--margin-type', default="dnm", type=str, help='dnm: dynamic margin, '
                                                                       'fnm: fixed margin(0.5), '
                                                                       'znm: no margin(0)',
                        choices=('dnm', 'fnm', 'znm'))


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