from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

class SingleDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc 
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = A_img.resize((self.opt.load_size//8, self.opt.load_size//8), Image.BICUBIC)
        A_img = A_img.resize((128, 128), Image.BICUBIC)
        A = self.transform(A_img)
        return {'LR': A, 'LR_paths': A_path}

    def __len__(self):
        return len(self.A_paths)
