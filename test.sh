# On CelebA Test set
python test.py --gpus 1 --model CMCNet --name CMCNET \
    --load_size 128 --dataset_name single --dataroot /Test_Celeba \
    --pretrain_model_path ./pretrain_models/... \
    --save_as_dir results_celeba/cmcnet