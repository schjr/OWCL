# -*- coding: utf-8 -*-
"""
@Time ： 2022/9/4 10:18
@Auth ： Ruijie Xu
@File ：TRSSL.py
@IDE ：PyCharm
"""
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import wandb
from utils.utils import AverageMeter, WeightEMA, interleave
from data.get_dataset import get_discover_datasets
from model.resnet_s import resnet18
from model.resnet import resnet50
import os
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
import numpy as np
from losses.margin import *
from utils.sinkhorn_knopp import SinkhornKnopp
import random
import torch.backends.cudnn as cudnn
from utils.uncr_util import uncr_generator
from utils.eval import split_cluster_acc_v1


def main(args):
    use_cuda = torch.cuda.is_available()

    # define dataset and dataloader
    dataset = get_discover_datasets(args.dataset, args)
    train_labeled_dataset, train_unlabeled_dataset, test_dataset_all, val_dataset, \
    uncr_dataset = dataset["train_label_dataset"], dataset["train_unlabel_dataset"], \
                   dataset["test_dataset"], dataset["val_dataset"], dataset["uncr_unlabel_dataset"]


    labeled_trainloader = torch.utils.data.DataLoader(
        train_labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    unlabeled_trainloader = torch.utils.data.DataLoader(
        train_unlabeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    uncr_loader = torch.utils.data.DataLoader(
        uncr_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader_all = torch.utils.data.DataLoader(
        test_dataset_all,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
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
        ema_model = resnet18(num_classes=args.num_classes)
        for param in ema_model.parameters():
            param.detach_()
    elif args.arch == "resnet50":
        model = resnet50(num_classes=args.num_classes)
        ema_model = resnet50(num_classes=args.num_classes)
        for param in ema_model.parameters():
            param.detach_()
    else:
        assert NotImplementedError("Only support resnet18, resnet50!!!")

    # Sinkorn-Knopp
    sinkhorn = SinkhornKnopp(args)

    model = model.cuda()
    ema_model = ema_model.cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        warmup_start_lr=0.001,
        eta_min=0.001,
    )
    ema_optimizer = WeightEMA(args, model, ema_model)

    # wandb
    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project, entity=args.entity, config=state, name=args.exp_name, dir="logs")
    temp_uncr = None
    # train
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('/model_path/')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # train
    for epoch in range(start_epoch, args.max_epochs):

        train_loss = train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, sinkhorn, epoch, use_cuda, wandb,temp_uncr)
        train_results = test(args, val_dataloader, ema_model, prefix="train")
        test_results = test(args, test_loader_all, ema_model, prefix="test")
        unmean = 0
        if args.uncr_freq > 0:
            if (epoch + 1) % args.uncr_freq == 0:
                temp_uncr = uncr_generator(args, uncr_loader, ema_model)
                unmean = np.mean(temp_uncr['uncr'])

        # log
        lr = np.around(optimizer.param_groups[0]["lr"], 4)
        print("--Comment-{}--Epoch-[{}/{}]--LR-[{}]--Loss-[{}]--All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".
              format(args.comment, epoch, args.max_epochs, lr, train_loss, train_results["train/all/avg"] * 100,
                     train_results["train/novel/avg"] * 100, train_results["train/seen/avg"] * 100))

        # call scheduler
        scheduler.step()
        results = {
            "lr": lr,
            "train_loss": train_loss,
            "mean_uncer": unmean
        }
        wandb.log(test_results)
        wandb.log(train_results)
        wandb.log(results)
        # save model
        if args.save_model:
            model_to_save = model.module if hasattr(model, "module") else model
            ema_model_to_save = ema_model.module if hasattr(ema_model, "module") else ema_model
            state_save = {
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = args.exp_name + '_' + str(epoch) + f'checkpoint.pth.tar'
            filepath = os.path.join(args.model_save_dir, filename)
            torch.save(state_save, filepath)


def train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, sinkhorn, epoch, use_cuda, logger, temp_uncr):
    losses = AverageMeter()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    for batch_idx in tqdm(range(args.train_iteration)):
        try:
            inputs_x, targets_x, idx_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, idx_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _, idx_u = unlabeled_train_iter.next()

        inputs_x = inputs_x[0] #这里注意v0
        batch_size = inputs_x.size(0)

        temp_u = args.temperature * np.ones_like(idx_u)
        temp_u = torch.tensor(temp_u)
        temp_x = args.temperature * np.ones_like(idx_x)
        temp_x = torch.tensor(temp_x)

        if temp_uncr:
            temp_u = np.array(temp_uncr['uncr'])[np.where(np.array(temp_uncr['index']) == np.array(idx_u)[:, None])[-1]]
            temp_u = torch.tensor(temp_u)


        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, args.num_classes).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
            temp_x, temp_u = temp_x.cuda(), temp_u.cuda()

        # normalize classifier weights
        with torch.no_grad():
            w = model.linear.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            w = w.cuda()
            model.linear.weight.copy_(w)


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u, _ = model(inputs_u)
            outputs_u2, _ = model(inputs_u2)


            # cross pseudo-labeling
            targets_u = sinkhorn(outputs_u2)
            targets_u2 = sinkhorn(outputs_u)

        # generate hard pseudo-labels for confident novel class samples
        targets_u_novel = targets_u[:, args.num_labeled_classes:]
        max_pred_novel, _ = torch.max(targets_u_novel, dim=-1)
        hard_novel_idx1 = torch.where(max_pred_novel >= args.threshold)[0]

        targets_u2_novel = targets_u2[:, args.num_labeled_classes:]
        max_pred2_novel, _ = torch.max(targets_u2_novel, dim=-1)
        hard_novel_idx2 = torch.where(max_pred2_novel >= args.threshold)[0]

        targets_u[hard_novel_idx1] = targets_u[hard_novel_idx1].ge(args.threshold).float()
        targets_u2[hard_novel_idx2] = targets_u2[hard_novel_idx2].ge(args.threshold).float()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u2], dim=0)
        all_temp = torch.cat([temp_x, temp_u, temp_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        temp_a, temp_b = all_temp, all_temp[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        mixed_temp = l * temp_a + (1 - l) * temp_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0]]

        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        logits = torch.cat((logits_x, logits_u), 0)

        #cross_entropy loss
        mixed_temp = torch.clamp(mixed_temp, min=1e-8)
        preds = F.softmax(logits / mixed_temp.unsqueeze(1), dim=1)
        preds = torch.clamp(preds, min=1e-8)
        preds = torch.log(preds)
        loss = -torch.mean(torch.sum(mixed_target * preds, dim=1))
        if torch.isnan(loss).sum() > 0:
            import ipdb;ipdb.set_trace()

        # record loss
        losses.update(loss.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        loss_results = {
            "loss": loss.clone(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(loss_results)

    return losses.avg


def test(args, test_loader, model, prefix):
    model.eval()
    preds = np.array([])
    gt_targets = np.array([])

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in tqdm(enumerate(test_loader)):
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs, _ = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            gt_targets = np.append(gt_targets, targets.cpu().numpy())
            preds = np.append(preds, max_idx.cpu().numpy())

    predictions = preds.astype(int)
    gt_targets = gt_targets.astype(int)
    results = split_cluster_acc_v1(gt_targets, predictions, num_seen=args.num_labeled_classes)
    log = {}
    for key, value in results.items():
        log[prefix+"/"+key + "/" + "avg"] = round(value, 4)
    return log

if __name__ == "__main__":
    parser = ArgumentParser(description='TRSSL Training')

    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--wdecay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--warmup-epochs', default=10, type=int, help='number of warmup epochs')

    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    # Method options
    parser.add_argument("--ratio", type=float, default=50, help="the percentage of labeled data")
    parser.add_argument('--train-iteration', type=int, default=1024, help='Number of iteration per epoch')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)
    parser.add_argument('--dataset', default='CIFAR100', help='dataset setting')

    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--imagenet-classes", default=100, type=int, help="number of ImageNet classes")
    parser.add_argument('--description', default="default_run", type=str, help='description of the experiment')
    parser.add_argument("--uncr-freq", default=1, type=int, help="frequency of generating uncertainty scores")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for hard pseudo-labeling")
    parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers")

    parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
    parser.add_argument("--project", default="OWSSL", type=str, help="wandb project")
    parser.add_argument("--entity", default="schubaru", type=str, help="wandb entity")
    parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
    parser.add_argument("--num_labeled_classes", default=50, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=50, type=int, help="number of unlab classes")
    parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
    parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
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
                                       f"unlabeled-{args.num_unlabeled_classes}-ratio-{int(args.ratio)}.pth")
    args.exp_name = "-".join(["discover", args.arch, args.dataset, args.comment,
                              str(args.num_labeled_classes), str(args.num_unlabeled_classes), str(int(args.ratio))])

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)

    main(args)



