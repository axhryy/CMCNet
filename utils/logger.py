import os
from collections import OrderedDict
import numpy as np
from .utils import mkdirs
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

class Logger():
    def __init__(self, opts):
        #%Y-%m-%d_%H_%M不能存在：
        time_stamp = '_{}'.format(datetime.now().strftime('%Y-%m-%d_%H_%M'))
        self.opts = opts
        self.log_dir = os.path.join(opts.log_dir, opts.name+time_stamp)
        self.phase_keys = ['train', 'val', 'test']
        self.iter_log = []
        self.epoch_log = OrderedDict() 
        self.set_mode(opts.phase)
        exist_log = None 
        for log_name in os.listdir(opts.log_dir):
            if opts.name in log_name:
                exist_log = log_name 
        if exist_log is not None: 
            old_dir = os.path.join(opts.log_dir, exist_log)
            archive_dir = os.path.join(opts.log_archive, exist_log) 
            shutil.move(old_dir, archive_dir)

        self.mk_log_file()

        self.writer = SummaryWriter(self.log_dir)

    def mk_log_file(self):
        mkdirs(self.log_dir)
        self.txt_files = OrderedDict()
        for i in self.phase_keys:
            self.txt_files[i] = os.path.join(self.log_dir, 'log_{}'.format(i))

    def set_mode(self, mode):
        self.mode = mode
        self.epoch_log[mode] = []

    def set_current_iter(self, cur_iter):
        self.cur_iter = cur_iter
        
    def record_losses(self, items):
        self.iter_log.append(items)
        for k, v in items.items():
            if 'loss' in k.lower():
                self.writer.add_scalar('loss/{}'.format(k), v, self.cur_iter)

    def record_scalar(self, items):
        for i in items.keys():
            self.writer.add_scalar('{}'.format(i), items[i], self.cur_iter)

    def record_image(self, visual_img, tag='ckpt_image'):
        self.writer.add_image(tag, visual_img, self.cur_iter, dataformats='HWC')
    
    def record_images(self, visuals, tag='ckpt_image'):
        for type_index, images in enumerate(visuals):
            for img_index, img in enumerate(images):
                assert img.dim() == 3, "图像维度应为 [C, H, W]"
                img_tag = f"{tag}/type_{type_index}_image_{img_index}"
                self.writer.add_image(img_tag, img, self.cur_iter)
        

    def record_text(self, tag, text):
        self.writer.add_text(tag, text) 

    def printIterSummary(self, epoch, cur_iters, total_it, timer):
        print(self.log_dir)
        msg = '{}\nIter: [{}]{:03d}/{:03d}\t\t'.format(
                timer.to_string(total_it - cur_iters), epoch, cur_iters, total_it)
        for k, v in self.iter_log[-1].items():
            msg += '{}: {:.6f}\t'.format(k, v) 
        print(msg + '\n')
        with open(self.txt_files[self.mode], 'a+') as f:
            f.write(msg + '\n')

    def close(self):
        self.writer.close()




