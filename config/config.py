# coding:utf-8
# Author : klein
# CONTACT: zhangchy2@shanghaitech.edu.cn
# SOFTWARE: PyCharm
# FILE : config.py
# DATE : 2022/8/27 15:35
# -----------------
# DATASET ROOTS
# -----------------
cifar_10_root = '/public/home/shijr/projects/data/ncd_data/NCD/cifar10'
cifar_100_root = '/public/home/shijr/projects/data/ncd_data/NCD/cifar100'
cub_root = '/public/home/shijr/projects/data/ncd_data/NCD/CUB_200_2011'
aircraft_root = '/public/home/shijr/projects/data/ncd_data/NCD/FGVCAircraft/fgvc-aircraft-2013b'
herbarium_dataroot = '/public/shijr/projects/data/ncd_data/NCD/herbarium_19/'
imagenet_root = '/public/home/shijr/projects/data/ncd_data/NCD/ILSVRC2012'
tinyimagenet_root = '/public/home/shijr/projects/data/ncd_data/NCD/tiny-imagenet-200'
scars_root = '/public/home/shijr/projects/data/ncd_data/NCD/stanford_cars'
oxfordiiitpet_root = '/public/home/shijr/projects/data/ncd_data/NCD/Oxford-IIIT-Pet'

# OSR Split dir
osr_split_dir = '/public/home/zhangchy2/workdir/Project/SSL/GCD/data/splits'

# -----------------
# OTHER PATHS
# -----------------
# dino_pretrain_path = '/public/home/zhangchy2/workdir/pretrained/dino/dino_vitbase16_pretrain.pth'
feature_extract_dir = '/public/home/shijr/projects/GCD/features'     # Extract features to this directory
exp_root = '/public/home/shijr/projects/GCD/experiments'          # All logs and checkpoints will be saved here