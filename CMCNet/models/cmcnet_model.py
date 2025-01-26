import torch
import torch.nn as nn
import torch.optim as optim
from models import loss 
from models import networks
from .base_model import BaseModel
from utils import utils
from models.cmcnet import CMCNet
import torchvision
from models.loss import PCPFeat,WTLoss,PerceptualLoss,FMLoss,CharbonnierLoss
# from torch.utils.tensorboard import SummaryWriter

class CMCNetModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        parser.add_argument('--scale_factor', type=int, default=8, help='upscale factor for CTCNet')
        parser.add_argument('--lambda_pix', type=float, default=1.0, help='weight for pixel loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netG = CMCNet() 
        self.netG = networks.define_network(opt, self.netG)

        self.model_names = ['G']
        self.load_model_names = ['G']
        self.loss_names = ['Total','Pix','per','wt'] 
        self.visual_names = ['img_LR', 'img_SR', 'img_HR']
        self.flag=0
        # self.writer = SummaryWriter('log')
        self.Wt=WTLoss()
        self.Per=PerceptualLoss(torch.nn.functional.mse_loss)
        self.pcpfeat = PCPFeat().cuda()
        self.FM=FMLoss()
        self.char=CharbonnierLoss()

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.optimizer_G]
    # def print_grad_hook(self,grad):
    #         print("woshi  __grad",grad)
    # def a(self,):
    #     for param in self.netG.parameters():
    #         param.register_hook(self.print_grad_hook)

    def load_pretrain_model(self,):
        print('Loading pretrained model', self.opt.pretrain_model_path)
        weight = torch.load(self.opt.pretrain_model_path)
        self.netG.module.load_state_dict(weight)
    
    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input['LR'].to(self.opt.data_device)
        self.img_HR = input['HR'].to(self.opt.data_device)

    def forward(self):
        self.img_SR = self.netG(self.img_LR) 
    #反向传播
    def backward_G(self):

        self.loss_wt=0
        self.loss_per=0
        # self.loss_wt=self.Wt(self.img_SR,self.img_HR)
        # resize_transform = torchvision.transforms.Resize((256, 256))
        # image_sr = resize_transform(self.img_SR)
        # image_hr = resize_transform(self.img_HR)
        # print("image_wt",self.loss_wt.requires_grad)
        # print("image_hr",image_hr.requires_grad)
        self.loss_Pix=self.criterionL1(self.img_SR, self.img_HR)
        # self.loss_per=self.Per(self.img_SR,self.img_HR)
        # print(".requires_grad ",self.img_HR.requires_grad )
        # self.loss_Pix=self.char(self.img_SR, self.img_HR)
        # Img_SR=self.pcpfeat(self.img_SR)
        # Img_HR=self.pcpfeat(self.img_HR)

        # self.loss_per=self.FM(Img_SR,Img_HR)
        # self.loss_Total =  self.criterionL1(self.img_SR, self.img_HR)+self.loss_wt*1.5+self.loss_per*0.005
        # self.loss_Total =  self.loss_Pix+self.loss_per*0.01+self.loss_wt*0.8
        # self.loss_Total =  self.loss_Pix
        self.loss_Total = self.loss_Pix +self.loss_per+self.loss_wt
        # self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * 0.8+perceptual_loss*0.2
        # print("criterionL1(self.img_SR, self.img_HR)",self.criterionL1(self.img_SR, self.img_HR),self.loss_wt*0.03,"per_loss",self.loss_per*0.01)
        # self.flag=self.flag+1
        # # print(self.flag)
        # self.writer.add_scalar('Loss/Loss1', self.criterionL1(self.img_SR, self.img_HR).item(), self.flag)
        # self.writer.add_scalar('Loss/Loss2', wt_loss.item(),self.flag)
        # self.writer.add_scalar('Loss/Total_Loss', self.loss_Pix.item(),self.flag)


        self.loss_Total.backward()
    
    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
    #将一些图像张量转换为可视化的图像
    # def get_current_visuals(self, size=128):
    #     out = []
    #     out.append(utils.tensor_to_numpy(self.img_LR))
    #     out.append(utils.tensor_to_numpy(self.img_SR))
    #     out.append(utils.tensor_to_numpy(self.img_HR))
    #     visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]
        
    #     return visual_imgs
    def get_current_visuals(self, size=128):
        stacked_tensor = torch.stack((self.img_LR,self.img_SR,self.img_HR), dim=0)#(3,3,3,128,128)
        return stacked_tensor
    






