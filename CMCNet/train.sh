# Train Code
python train.py --gpus 1 --name wfen --model wfen \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /path/to/datasets/CelebA --dataset_name celeba --batch_size 10 --total_epochs 20 \
    --visual_freq 100 --print_freq 50 --save_latest_freq 1000
