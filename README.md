## CMCNet: Enhancing Face Image Super-Resolution through CNN-Mamba Collaboration

## Installation and Requirements 
I have trained and tested the codes on
- Ubuntu 20.04
- CUDA 11.1  
- Python 3.8

## Getting Started
Download Our Pretrain Models and Test Dataset. Additionally, we offer our FSR results in orginal paper.
#### Note：Test results are slightly different from the original paper because the model weights were obtained by re-training after organizing our codes.
- [Pretrain_Models](https://drive.google.com/file/d/1YTVf_xK1Ua21zfEUfw3tDQTmbsF3Ld4k/view?usp=drive_link)  
- [Test_Datasets](https://drive.google.com/file/d/1EW-DZvmIPzMQcYrrwspODoKgFA0oBeR2/view?usp=drive_link)
- [FSR_Results_in_Orginal_Paper](https://drive.google.com/file/d/136DlSB1FvI8timRgDL1WRIx8JyKvtwdO/view?usp=drive_link)

### Train the Model
The commands used to train the released models are provided in script `train.sh`. Here are some train tips:
- You should download [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to train CMCNet. Please change the `--dataroot` to the path where your training images are stored.  
- To train CMCNet, we use OpenFace to detect 68 facial landmarks to crop the face images, which were then resized to 128×128 pixels as HR images without applying any alignment techniques.
- These HR images are downsampled to 16×16 and 32×32 pixels using bicubic interpolation and used as LR input for ×8 FSR and ×4 FSR tasks,
- Please change the `--name` option for different experiments. Tensorboard records with the same name will be moved to `checkpoints/log_archive`, and the weight directory will only store weight history of latest experiment with the same name.

```
# Train Code
python train.py --gpus 1 --name wfen --model wfen \
    --Gnorm "bn" --lr 0.0002 --beta1 0.9 --scale_factor 8 --load_size 128 \
    --dataroot /path/to/datasets/CelebA --dataset_name celeba --batch_size 10 --total_epochs 20 \
    --visual_freq 100 --print_freq 50 --save_latest_freq 1000
```


### Test with Pretrained Models
```
# On CelebAx8 Test set
python test.py --gpus 1 --model CMCNet --name CMCNET \
    --load_size 128 --dataset_name single --dataroot /Test_Celeba \
    --pretrain_model_path ./pretrain_models/wfen/wfen_best.pth \
    --save_as_dir results_celeba/cmcnet
```

```
# On Helenx8 Test set
python test.py --gpus 1 --model wfen --name CMCNET \
    --load_size 128 --dataset_name single --dataroot /Test_Helen\
    --pretrain_model_path ./pretrain_models/wfen/wfen_best.pth \
    --save_as_dir results_helen/cmcnet
```

## Acknowledgements
This code is built on [Face-SPARNet](https://github.com/chaofengc/Face-SPARNet). We thank the authors for sharing their codes.

