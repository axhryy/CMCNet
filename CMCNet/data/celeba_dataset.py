import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

from data.base_dataset import BaseDataset


class CelebADataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.shuffle = True if opt.isTrain else False
        # load_size 128 scale_factor 8
        self.lr_size = opt.load_size // opt.scale_factor
        self.hr_size = opt.load_size
        # image/
        self.img_dir = opt.dataroot
        self.img_names = self.get_img_names()

        self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                Scale((1.0, 1.3), opt.load_size) 
                ])

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
    """
    img_names = [x for x in os.listdir(self.img_dir)]：使用os.listdir函数获取指定目录self.img_dir下的所有文件名，
    并将它们存储在img_names列表中。这里使用了列表推导式来简洁地生成列表。
    if self.shuffle:：检查是否需要对图像文件名进行随机打乱。self.shuffle是一个布尔值，表示是否进行打乱操作。
    random.shuffle(img_names)：如果需要打乱图像文件名列表，则使用random.shuffle函数对img_names列表进行随机打乱操作。
    这样可以改变图像文件名的顺序，使其在训练过程中随机化。
    """
    def get_img_names(self,):
        img_names = [x for x in os.listdir(self.img_dir)] 
        if self.shuffle:
            random.shuffle(img_names)
        "列表"
        return img_names

    def __len__(self,):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])

        hr_img = Image.open(img_path).convert('RGB')
        hr_img = self.aug(hr_img)

        # downsample and upsample to get the LR image
        lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC) 
        lr_img_up = lr_img.resize((self.hr_size, self.hr_size), Image.BICUBIC) 

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img_up)
        # print("hr is",hr_tensor.size(),img_path,"lr is",lr_tensor.size())
        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path}

class Scale():
    """
    Random scale the image and pad to the same size if needed.
    ---------------
    # Args:
        factor: tuple input, max and min scale factor.        
    """
    def __init__(self, factor, size):
        self.factor = factor 
        rc_scale = (2 - factor[1], 1)
        self.size   = (size, size)
        self.rc_scale = rc_scale
        self.ratio = (3. / 4., 4. / 3.) 
        self.resize_crop = transforms.RandomResizedCrop(size, rc_scale)

    def __call__(self, img):
        scale_factor = random.random() * (self.factor[1] - self.factor[0]) + self.factor[0]  
        w, h = img.size
        sw, sh = int(w*scale_factor), int(h*scale_factor)
        scaled_img = tf.resize(img, (sh, sw))
        if sw > w:
            i, j, h, w = self.resize_crop.get_params(img, self.rc_scale, self.ratio)
            scaled_img = tf.resized_crop(img, i, j, h, w, self.size, Image.BICUBIC) 
        elif sw < w:
            lp = (w - sw) // 2
            tp = (h - sh) // 2 
            padding = (lp, tp, w - sw - lp, h - sh - tp) 
            scaled_img = tf.pad(scaled_img, padding)
        return scaled_img 

