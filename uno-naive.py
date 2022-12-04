# Continual Learning Setting of UNO
# Currently we assume that there is a two stage setting
# In the first stage, labelled classes are learned through pretraining
# In the second stage, unlabelled classes are learned there
# Previously called uno-con4.py

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import contextlib
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from data.get_dataset import get_raw_datasets
from model.MultiHeadResNet import MultiHeadResNet
import os
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from utils.eval import split_cluster_acc_v1, split_cluster_acc_v2
import numpy as np
from losses.sinkhorn_knopp import SinkhornKnopp
from utils.utils import model_statistics
import ipdb


def main(args):
    device = torch.device('cuda:0')

    # define dataset and dataloader
    dataset = get_raw_datasets(args.dataset, args)
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

    original_model = MultiHeadResNet(
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
        original_model.load_state_dict(updated_state_dict, strict=False)
    else:
        print("Resnet50 or Resnet18 pretrained!")
        state_dict = torch.load(args.pretrained, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        model.load_state_dict(state_dict, strict=False)
        original_model.load_state_dict(state_dict, strict=False)

    if args.arch == "resnet18":
        print("Fine tuning feature after layer4.")
        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False
        for name, param in original_model.named_parameters():
            param.requires_grad = False

    model = model.cuda()
    original_model = original_model.cuda()
    model_statistics(model)

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
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('model_path/...........')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch, args.max_epochs):
        loss_per_head = train(args, model, original_model, sk, loss_per_head, train_dataloader, optimizer, scheduler, wandb)
        best_head = torch.argmin(loss_per_head)
        test_results = test(args, model, original_model, test_dataloader, best_head, prefix="test")
        train_results = test(args, model, original_model, val_dataloader, best_head, prefix="train")

        scheduler.step()

        # log
        lr = np.around(optimizer.param_groups[0]["lr"], 4)
        print("--Comment-{}--Epoch-[{}/{}]--LR-[{}]--All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".
              format(args.comment, epoch, args.max_epochs, lr, train_results["train/all/avg"]*100,
                     train_results["train/novel/avg"]*100, train_results["train/seen/avg"]*100))

        wandb.log(train_results)
        wandb.log(test_results)

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


def cross_entropy_loss(preds, targets, temperature):
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def swapped_prediction(args, logits, targets):
    loss = 0
    for view in range(args.num_large_crops):
        for other_view in np.delete(range(args.num_crops), view):
            loss += cross_entropy_loss(logits[other_view], targets[view], temperature=args.temperature)
    return loss / (args.num_large_crops * (args.num_crops - 1))


def train(args, model, original_model, sk, loss_per_head, train_dataloader, optimizer, scheduler, logger):
    model.train()
    bar = tqdm(train_dataloader)
    nlc = args.num_labeled_classes
    scaler = GradScaler()
    amp_cm = autocast() if args.amp else contextlib.nullcontext()

    for batch in bar:
        optimizer.zero_grad()

        # unpack multi-view images
        views_unlab, labels_unlab, _ = batch
        images = views_unlab
        images = [image.cuda() for image in images]
        # ipdb.set_trace()

        # normalize prototypes
        model.normalize_prototypes()

        with amp_cm:
            # forward
            outputs = model(images)
            # original_outputs = original_model(images)
            #
            # # gather outputs
            # original_outputs["logits_lab"] = (
            #     original_outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1)
            # )
            # logits = torch.cat([original_outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
            # # logits = outputs["logits_unlab"]
            #
            # # create targets
            # original_targets = torch.zeros_like(original_outputs["logits_lab"])
            # unseen_targets = torch.zeros_like(outputs["logits_unlab"])
            #
            # # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
            # for v in range(args.num_large_crops):
            #     for h in range(args.num_heads):
            #         unseen_targets[v, h] = sk(outputs["logits_unlab"][v, h]).type_as(unseen_targets)
            #
            # targets = torch.cat([original_targets, unseen_targets], dim=-1)
            logits = outputs["logits_unlab"]
            targets = torch.zeros_like(logits)
            for v in range(args.num_large_crops):
                for h in range(args.num_heads):
                    targets[v, h] = sk(logits[v, h]).type_as(targets)

            # compute swapped prediction loss
            loss_cluster = swapped_prediction(args, logits, targets)
            # update best head tracker
            loss_per_head += loss_cluster.clone().detach()
            loss_cluster = loss_cluster.mean()
            loss = loss_cluster

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
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(results)

    return loss_per_head


@torch.no_grad()
def test(args, model, original_model, val_dataloader, best_head, prefix):
    model.eval()
    all_labels = None
    all_preds = None
    with torch.no_grad():
        for batch in val_dataloader:
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            original_outputs = original_model(images)

            preds_inc = torch.cat(
                [
                    original_outputs["logits_lab"].unsqueeze(0).expand(args.num_heads, -1, -1),
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
        _res = split_cluster_acc_v2(all_labels, all_preds[head], num_seen=args.num_labeled_classes)
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
    parser.add_argument("--dataset", default="CIFAR100Con", type=str, help="dataset")
    parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--num_workers", default=10, type=int, help="number of workers")

    parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument('--amp', default=False, action="store_true", help='use mixed precision training or not')
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--max_epochs", default=10, type=int, help="warmup epochs")

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
    parser.add_argument("--project", default="OWSSL", type=str, help="wandb project")
    parser.add_argument("--entity", default="schubaru", type=str, help="wandb entity")
    parser.add_argument("--offline", default=True, action="store_true", help="disable wandb")
    parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labelled classes")
    parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlabelled classes")
    parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
    parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
    parser.add_argument("--ss_pretrained", default=False, action="store_true", help="self-supervised pretrain")
    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--val_ratio", type=float, default=10, help="the percentage of validation data")
    parser.add_argument("--regenerate", default=False, action="store_true", help="whether to generate data again")
    parser.add_argument("--resume", default=False, action="store_true", help="whether to use old model")
    parser.add_argument("--save-model", default=False, action="store_true", help="whether to save model")

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
                                       f"unlabeled-{args.num_unlabeled_classes}-val_ratio-{int(args.val_ratio)}.pth")
    args.exp_name = "-".join(["discover", args.arch, args.dataset, args.comment,
                              str(args.num_labeled_classes), str(args.num_unlabeled_classes), str(int(args.val_ratio))])
    main(args)
    #  python uno.py --dataset CIFAR100Con --gpus 0 --max_epochs 200 --batch_size 256 --num_labeled_classes 50
    #  --num_unlabeled_classes 50 --pretrained checkpoints/pretrained/supervised-CIFAR100-resnet18-labeled-50-unlabeled-50-ratio-50.pth
    #  --num_heads 4 --comment UNO-amp --offline  --multicrop --amp
    #

