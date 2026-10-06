# TESS CIFAR10

# alpha_post = -1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --arch cifar_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR10_VGG_TESS_apo_neg1 --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 -1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 0
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --arch cifar_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR10_VGG_TESS_apo_zero --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam


# alpha_post = 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10 --arch cifar_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR10_VGG_TESS_apo_pos1 --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam



# TESS CIFAR100

# alpha_post = -1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR100 --arch cifar100_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR100_VGG_TESS_apo_neg1 --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 -1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR100" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 0
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR100 --arch cifar100_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR100_VGG_TESS_apo_zero --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR100" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR100 --arch cifar100_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR100_VGG_TESS_apo_pos1 --trials 1 --epochs 200 --batch-size 128 --val-batch-size 64 --print-freq 20 --delay-ls 6 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "CIFAR100" --training-mode tess --loss "CE" --wn --optimizer Adam



# TESS DVSCIFAR10

# alpha_post = -1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR10DVS_VGG_TESS_apo_neg1 --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --delay-ls 10 --factors-stdp 0.2 0.5 -1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 0
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR10DVS_VGG_TESS_apo_zero --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --delay-ls 10 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset CIFAR10DVS --arch dvscifar10_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/CIFAR10DVS_VGG_TESS_apo_pos1 --trials 1 --epochs 200 --batch-size 64 --val-batch-size 64 --print-freq 20 --delay-ls 10 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSCIFAR10" --training-mode tess --loss "CE" --wn --optimizer Adam


# TESS DVS Gesture

# alpha_post = -1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset DVSGesture --arch dvs_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/DVSGesture_TESS_apo_neg1 --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --delay-ls 20 --factors-stdp 0.2 0.5 -1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 0
CUDA_VISIBLE_DEVICES=0 python main.py --dataset DVSGesture --arch dvs_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/DVSGesture_TESS_apo_zero --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --delay-ls 20 --factors-stdp 0.2 0.5 0 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode tess --loss "CE" --wn --optimizer Adam

# alpha_post = 1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset DVSGesture --arch dvs_tessvgg_model --data-path ~/Datasets --save-path ./experiments/alpha_post_test/DVSGesture_TESS_apo_pos1 --trials 1 --epochs 200 --batch-size 16 --val-batch-size 16 --print-freq 20 --delay-ls 20 --factors-stdp 0.2 0.5 1 1 --pooling MAX --scheduler 100 --lr 0.001 --lr-conv 0.001 --experiment-name "DVSGesture" --training-mode tess --loss "CE" --wn --optimizer Adam