## CMCNet: Enhancing Face Image Super-Resolution through CNN-Mamba Collaboration

## Installation and Requirements 
I have trained and tested the codes on
- Ubuntu 20.04
- CUDA 11.8  
- Python 3.8

### Pretrained models and test results
The **pretrained models** and **test results** are being reorganized.

### Train the Model
The commands used to train the released models are provided in script `train.sh`. Here are some train tips:
- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train CMCNet. Please change the `--dataroot` to the path where your training images are stored.  
- To train CMCNet, we use OpenFace to detect 68 facial landmarks to crop the face images, which were then resized to 128×128 pixels as HR images without applying any alignment techniques.
- These HR images are downsampled to 16×16 and 32×32 pixels using bicubic interpolation and used as LR input for ×8 FSR and ×4 FSR tasks,
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `checkpoints/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.

```
# Train Code
python train.py --gpus 1 --name CMCNET --model cmcnet \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /CelebA --dataset_name celeba --batch_size 10 --total_epochs 20 \
    --visual_freq 100 --print_freq 50 --save_latest_freq 1000
```


### Test with Pretrained Models
```
# On CelebAx8 Test set
python test.py --gpus 1 --model cmcnet --name CMCNET \
    --load_size 128 --dataset_name single --dataroot /Test_Celeba \
    --pretrain_model_path ./checkpoints/CMCNET/.ph \
    --save_as_dir results_celeba/cmcnet
```

```
# On Helenx8 Test set
python test.py --gpus 1 --model cmcnet --name CMCNET \
    --load_size 128 --dataset_name single --dataroot /Test_Helen\
    --pretrain_model_path ./checkpoints/CMCNET/.ph \
    --save_as_dir results_helen/cmcnet
```

## Acknowledgements
This code is built on [Face-SPARNet](https://github.com/chaofengc/Face-SPARNet). We thank the authors for sharing their codes.

