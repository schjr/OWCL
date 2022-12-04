# Running Scripts

    salloc -N 1 -n 16 -p normal --gres=gpu:NVIDIATITANV:1 --time 0-8:0:0

---

    python uno.py --dataset CIFAR100 --gpus 0 --max_epochs 200 --batch_size 256 --num_labeled_classes 50 --num_unlabeled_classes 50 --pretrained checkpoints/pretrained/supervised-CIFAR100-resnet18-labeled-50-unlabeled-50-ratio-50.pth --num_heads 4 --comment UNO-amp --offline  --multicrop --amp 

---

    python MI.py --dataset CIFAR100 --gpus 0 --max_epochs 200 --batch_size 512 --num_labeled_classes 50 --num_unlabeled_classes 50 --pretrained checkpoints/pretrained/supervised-CIFAR100-resnet18-labeled-50-unlabeled-50-ratio-50.pth --num_heads 4 --comment MI-512 --offline --multicrop --save-model --amp

    export TMPDIR=/public/home/shijr/tmp
