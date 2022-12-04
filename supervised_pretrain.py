# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : supervised_pretrain.py
# DATE : 2022/8/27 17:47
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from data.get_dataset import get_supervised_datasets
from model.MultiHeadResNet import MultiHeadResNet
import os
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from utils.eval import accuracy
import numpy as np


def main(args):
    # define dataset and dataloader
    train_dataset, test_dataset = get_supervised_datasets(args.dataset, args)
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
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

    # define model
    model = MultiHeadResNet(arch=args.arch,
                            low_res="CIFAR" in args.dataset or "tiny" in args.dataset,
                            num_labeled=args.num_labeled_classes,
                            num_unlabeled=args.num_unlabeled_classes,
                            num_heads=None)
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum_opt,
        weight_decay=args.weight_decay_opt,
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=args.min_lr,
        eta_min=args.min_lr,
    )

    # wandb
    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project, entity=args.entity, config=state, name=args.exp_name, dir="logs")

    # train
    for epoch in range(args.max_epochs):
        train(args, model, train_dataloader, optimizer, scheduler, wandb)
        results = test(args, model, test_dataloader)
        scheduler.step()

        # log
        acc = np.around(results["prec1"][0].detach().cpu().numpy(), 2)
        lr = np.around(optimizer.param_groups[0]["lr"], 4)
        print(f"--Epoch-[{epoch}]--LR-[{lr}]--Acc-[{acc}]--")
        wandb.log(results)

    # save model
    torch.save(model.state_dict(), args.model_save_dir)


def train(args, model, train_dataloader, optimizer, scheduler, logger):
    model.train()
    bar = tqdm(train_dataloader)
    for batch in bar:
        images, labels, _ = batch
        images = [image.cuda() for image in images]
        labels = labels.cuda()

        model.normalize_prototypes()

        outputs = model(images)

        loss_supervised = torch.stack(
            [F.cross_entropy(o / args.temperature, labels) for o in outputs["logits_lab"]]
        ).mean()

        optimizer.zero_grad()
        loss_supervised.backward()
        optimizer.step()

        bar.set_postfix({"loss": "{:.2f}".format(loss_supervised.detach().cpu().numpy())})
        results = {
            "loss_supervised": loss_supervised.clone(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(results)


def test(args, model, val_dataloader):
    model.eval()
    all_labels = None
    all_preds = None
    with torch.no_grad():
        for batch in val_dataloader:
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)["logits_lab"]

            if all_labels is None:
                all_labels = labels
                all_preds = outputs
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_preds = torch.cat([all_preds, outputs], dim=0)

    prec1 = accuracy(all_preds, all_labels)
    return {"prec1": prec1}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
    parser.add_argument("--download", default=False, action="store_true", help="whether to download")
    # parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--num_workers", default=5, type=int, help="number of workers")
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--max_epochs", default=100, type=int, help="max epochs")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
    parser.add_argument("--project", default="OWSSL", type=str, help="wandb project")
    parser.add_argument("--entity", default="schubaru", type=str, help="wandb entity")
    parser.add_argument("--offline", default=True, action="store_true", help="disable wandb")
    parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
    parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
    parser.add_argument("--pretrained", type=str, default=None, help="pretrained checkpoint path")
    parser.add_argument("--ratio", type=float, default=50, help="the percentage of labeled data")
    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")

    args = parser.parse_args()
    os.environ["WANDB_API_KEY"] = "8a2383b05d272ebfdf491e9e9a9f616a24323853"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.model_save_dir = os.path.join(args.checkpoint_dir, "pretrained",
                                       f"supervised-{args.dataset}-{args.arch}-labeled-{args.num_labeled_classes}-"
                                       f"unlabeled-{args.num_unlabeled_classes}-ratio-{int(args.ratio)}.pth")
    args.exp_name = "-".join(["pretrain", args.arch, args.dataset, args.comment,
                              str(args.num_labeled_classes), str(args.num_unlabeled_classes), str(int(args.ratio))])
    main(args)
