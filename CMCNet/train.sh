# export CUDA_VISIBLE_DEVICES=$1
# =================================================================================
# Train CTCNet
# =================================================================================
#train erf
python train.py --gpus 1 --name CTCNet_S16_V4_Attn2D --model ctcnet --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 4 --load_size 128 --dataroot "/root/autodl-tmp/CMCNet_Train_179244" --dataset_name celeba --batch_size 5 --total_epochs 150 --visual_freq 150 --print_freq 50 --save_latest_freq 1000

python train.py --gpus 1 --name CTCNet_S16_V4_Attn2D --model ctcnet --Gnorm "bn" --lr 0.00015 --beta1 0.9 --scale_factor 8 --load_size 128 --dataroot "/root/autodl-fs/CTCNet-main/image_test" --dataset_name celeba --batch_size 8 --total_epochs 100 --visual_freq 100 --print_freq 50 --save_latest_freq 1000 --continue_train

python train.py --gpus 1 --name CTCNet_S16_V4_Attn2D --model ctcnet --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 4 --load_size 128 --dataroot "/root/autodl-tmp/CMCNet_Train_179244" --dataset_name celeba --batch_size 6 --total_epochs 20 --visual_freq 150 --print_freq 50 --save_latest_freq 1000