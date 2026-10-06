

# TESS DVSCIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_tessvgg_model --data-path ~/Datasets --save-path ./experiments/CIFAR10DVS_VGG_TESS_ --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --delay-ls 10 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam

# BPTT DVSCIFAR10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_vgg_bptt --data-path ~/Datasets --save-path ./experiments/CIFAR10DVS_VGG_BPTT_ --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode bptt --loss "CE" --wn --optimizer Adam