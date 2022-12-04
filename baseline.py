# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : baseline.py
# DATE : 2022/8/28 10:04


import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from torch.cuda.amp import autocast, GradScaler
import contextlib
from data.get_dataset import get_discover_datasets
from model.MultiHeadResNet import MultiHeadResNet
import os
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from utils.eval import split_cluster_acc_v1
import numpy as np
from losses.sinkhorn_knopp import SinkhornKnopp


def main(args):
    device = torch.device('cuda')

    # define dataset and dataloader
    dataset = get_discover_datasets(args.dataset, args)
    train_dataset, test_dataset, val_dataset = dataset["train_dataset"], dataset["test_dataset"], dataset["val_dataset"]

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
    val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    # define model
    model = MultiHeadResNet(
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

    # simclr pretrain
    if args.ss_pretrained:
        print("SimClr pretrained!")
        state_dict = torch.load(args.pretrained)
        updated_state_dict = {}
        for key, value in state_dict.items():
            updated_state_dict[f"encoder.{key}"] = value
        model.load_state_dict(updated_state_dict, strict=False)
    else:
        print("Resnet50 or Resnet18 pretrained!")
        state_dict = torch.load(args.pretrained, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        model.load_state_dict(state_dict, strict=False)

    if args.arch != "resnet18":
        print("Fine tuning feature after layer4.")
        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False

    model = model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # opt and scheduler
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

    sk = SinkhornKnopp()
    loss_per_head = torch.zeros(args.num_heads).cuda()
    # train
    for epoch in range(args.max_epochs):
        loss_per_head = train(args, model, sk, loss_per_head, train_dataloader, optimizer, scheduler, wandb)
        best_head = torch.argmin(loss_per_head)
        test_results = test(args, model, test_dataloader, best_head, prefix="test")
        train_results = test(args, model, val_dataloader, best_head, prefix="train")

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


def cross_entropy_loss(preds, targets, temperature):
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def swapped_prediction(args, logits, targets):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            loss += cross_entropy_loss(logits[other_view], targets[view], temperature=args.temperature)
    return loss / (args.num_large_crops * (args.num_crops - 1))


def train(args, model, sk, loss_per_head, train_dataloader, optimizer, scheduler, logger):
    model.train()
    bar = tqdm(train_dataloader)
    nlc = args.num_labeled_classes
    scaler = GradScaler()
    amp_cm = autocast() if args.amp else contextlib.nullcontext()

    for batch in bar:
        optimizer.zero_grad()

        # unpack multi-view images
        views_lab, labels_lab, _, views_unlab, labels_unlab, _ = batch
        images = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
        labels = torch.cat([labels_lab, labels_unlab])

        images = [image.cuda() for image in images]
        labels = labels.cuda()

        mask_lab = torch.zeros_like(labels)
        mask_lab[:args.batch_size] = 1
        mask_lab = mask_lab.to(torch.bool)
        labels = labels.long()

        # normalize prototypes
        model.normalize_prototypes()

        model.eval()
        with torch.no_grad():
            # forward
            outputs = model(images)
        model.train()

        # gather outputs
        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)

        # create targets
        targets_lab = F.one_hot(labels[mask_lab], num_classes=args.num_labeled_classes).float()

        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(args.num_large_crops):
            for h in range(args.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab] = sk(logits[v, h, ~mask_lab]).type_as(targets)
                targets_over[v, h, ~mask_lab] = sk(logits_over[v, h, ~mask_lab]).type_as(targets)

        r = np.random.beta(args.mix_alpha, args.mix_alpha)
        # rr = 1
        idx = torch.randperm(len(images[0]))

        mixed_input = [r * image + (1 - r) * image[idx] for image in images]
        mixed_target = [r * target + (1 - r) * target[:, idx] for target in targets]
        mixed_target_over = [r * target_over + (1 - r) * target_over[:, idx] for target_over in targets_over]

        with amp_cm:
            mix_outputs = model(mixed_input)
            mix_outputs["logits_lab"] = (
                    mix_outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1)
            )
            mix_logits = torch.cat([mix_outputs["logits_lab"], mix_outputs["logits_unlab"]], dim=-1)
            mix_logits_over = torch.cat([mix_outputs["logits_lab"], mix_outputs["logits_unlab_over"]], dim=-1)

            loss_cluster = swapped_prediction(args, mix_logits, mixed_target)
            loss_overcluster = swapped_prediction(args, mix_logits_over, mixed_target_over)

            # update best head tracker
            loss_per_head += loss_cluster.clone().detach()

            loss_cluster = loss_cluster.mean()
            loss_overcluster = loss_overcluster.mean()

            # ignore overcluster
            loss = (loss_cluster + loss_overcluster) / 2

        # loss = loss_cluster
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
            "loss_cluster": loss_cluster.clone(),
            "loss_overcluster": loss_overcluster.clone(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(results)

    return loss_per_head


@torch.no_grad()
def test(args, model, val_dataloader, best_head, prefix):
    model.eval()
    all_labels = None
    all_preds = None
    with torch.no_grad():
        for batch in val_dataloader:
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)

            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(args.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )

            preds_inc = preds_inc.max(dim=-1)[1]

            if all_labels is None:
                all_labels = labels
                all_preds = preds_inc
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_preds = torch.cat([all_preds, preds_inc], dim=1)

    all_labels = all_labels.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()

    results = {}
    for head in range(args.num_heads):
        _res = split_cluster_acc_v1(all_labels, all_preds[head], num_seen=args.num_labeled_classes)
        for key, value in _res.items():
            if key in results.keys():
                results[key].append(value)
            else:
                results[key] = [value]

    log = {}
    for key, value in results.items():
        log[prefix + "/" + key + "/" + "avg"] = round(sum(value) / len(value), 4)
        log[prefix + "/" + key + "/" + "best"] = round(value[best_head], 4)

    return log


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
    parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--num_workers", default=10, type=int, help="number of workers")

    parser.add_argument('--amp', default=False, action="store_true", help='use mixed precision training or not')
    parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--max_epochs", default=200, type=int, help="max epochs")

    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
    parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
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
    parser.add_argument("--ss_pretrained", default=False, action="store_true", help="self-supervised pretrain")
    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--ratio", type=float, default=50, help="the percentage of labeled data")
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")
    parser.add_argument("--mix_alpha", default=0.75, type=float, help="the hyper-parameter of alpha")

    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args.num_over_classes = args.num_labeled_classes + args.num_unlabeled_classes*args.overcluster_factor

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    # os setting
    os.environ["WANDB_API_KEY"] = "8a2383b05d272ebfdf491e9e9a9f616a24323853"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.model_save_dir = os.path.join(args.checkpoint_dir, "discover",
                                       f"supervised-{args.dataset}-{args.arch}-labeled-{args.num_labeled_classes}-"
                                       f"unlabeled-{args.num_unlabeled_classes}-ratio-{int(args.ratio)}.pth")
    args.exp_name = "-".join(["discover", args.arch, args.dataset, args.comment,
                              str(args.num_labeled_classes), str(args.num_unlabeled_classes), str(int(args.ratio))])
    main(args)


