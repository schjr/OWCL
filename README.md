# Feature alignment for open-world semi-supervised learning

Training Features
- [ ] Distributed Training
- [ ] Amp training

Supported Algorithms
- [ ] GCD
- [ ] ORCA
- [ ] RankStat
- [ ] DTC


## Training scripts
uno
```python
# train-all: 55.81, novel: 48.06, seen: 75.07. test-all: 60.76, novel: 47.38, seen: 76.87
 python uno.py --dataset CIFAR100 --gpus 0 --max_epochs 200 --batch_size 256 --num_labeled_classes 50 --num_unlabeled_classes 50 --pretrained checkpoints/pretrained/supervised-CIFAR100-resnet18-labeled-50-unlabeled-50-ratio-50.pth --num_heads 4 --comment UNO-amp --offline  --multicrop --amp 
```

mi
```python
# train-all: 61.29, novel: 54.14, seen: 78.11. test-all: 64.52, novel: 54.76, seen: 75.76
python MI.py --dataset CIFAR100 --gpus 0 --max_epochs 200 --batch_size 256 --num_labeled_classes 50 --num_unlabeled_classes 50 --pretrained checkpoints/pretrained/supervised-CIFAR100-resnet18-labeled-50-unlabeled-50-ratio-50.pth --num_heads 4 --comment MI --offline --multicrop
```

orca
```python
# 
python orca.py --dataset CIFAR100 --num_labeled_classes 50 --ratio 50 --batch-size 512 --name orca --pretrained checkpoints/pretrained/simclr_imagenet_100.pth.tar --gpus 0 --num_unlabeled_classes 50 --max_epochs 200 --margin-type fnm --offline
```

## dataset

| dataset | CUB200 | Stanford |
| :---: | :---: | :---: |
| seen | 100 | 98 |
| unseen | 100 | 98 |