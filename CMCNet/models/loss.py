import torch
from torchvision import models
from utils import utils
from torch import nn, autograd
from torch.nn import functional as F
import torch
import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pytorch_wavelets as pw
from torchvision.transforms import ToTensor,Resize
import torchvision
from torchvision.models import VGG19_Weights 

class PCPFeat(torch.nn.Module):
    """
    Features used to calculate Perceptual Loss based on ResNet50 features.
    Input: (B, C, H, W), RGB, [0, 1]
    """
    def __init__(self,):
        super(PCPFeat, self).__init__()
        
        self.model = models.vgg19(pretrained=True)
        self.build_vgg_layers()

        # self.model.load_state_dict(torch.load(weight_path))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def build_vgg_layers(self):
        vgg_pretrained_features = self.model.features
        self.features = []
        feature_layers = [0, 3, 8, 17, 26, 35]
        for i in range(len(feature_layers)-1): 
            module_layers = torch.nn.Sequential() 
            for j in range(feature_layers[i], feature_layers[i+1]):
                module_layers.add_module(str(j), vgg_pretrained_features[j])
            self.features.append(module_layers)
        self.features = torch.nn.ModuleList(self.features)

    def preprocess(self, x):
        x = (x + 1) / 2
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x)
        std  = torch.Tensor([0.229, 0.224, 0.225]).to(x)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        x = (x - mean) / std
        if x.shape[3] < 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x
    def forward(self, x):
        x = self.preprocess(x)
        
        features = []
        for m in self.features:
            x = m(x)
            features.append(x)
        return features


class PCPLoss(torch.nn.Module):
    """Perceptual Loss.
    """
    def __init__(self, 
            opt, 
            layer=5,
            model='vgg',
            ):
        super(PCPLoss, self).__init__()

        self.crit = torch.nn.L1Loss()
        #  self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1, 1, 1, 1, 1]
    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf, w in zip(x_feats, y_feats, self.weights): 
            loss = loss + self.crit(xf, yf.detach()) * w
        return loss 


class FMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.crit  = torch.nn.L1Loss()

    def forward(self, x_feats, y_feats):
        loss = 0
        for xf, yf in zip(x_feats, y_feats):
            loss = loss + self.crit(xf, yf.detach()) 
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            pass
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode in ['softwgan']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    loss = nn.ReLU()(1 - prediction).mean()
                else:
                    loss = nn.ReLU()(1 + prediction).mean() 
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss  = - prediction.mean()
            return loss

        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'softwgan':
            if target_is_real:
                loss = F.softplus(-prediction).mean()
            else:
                loss = F.softplus(prediction).mean()
        return loss
# class WTLoss(nn.Module):
#     def __init__(self):
#         super(WTLoss, self).__init__()
#         self.dwt = pw.DWTForward(wave='haar', mode='symmetric').cuda()   # 使用 pytorch_wavelets

#     def wavelet_transform(self, image_tensor):
#         Yl, Yh = self.dwt(image_tensor) # Yl 是低频部分，Yh 是包含高频部分的列表
#         cH, cV, cD = Yh[0][:, 0, :, :], Yh[0][:, 1, :, :], Yh[0][:, 2, :, :]  # 提取高频部分
#         return Yl, cH, cV, cD

#     def forward(self, img_SR, img_HR):
#         _, cH_SR, cV_SR, cD_SR = self.wavelet_transform(img_SR)
#         _, cH_HR, cV_HR, cD_HR = self.wavelet_transform(img_HR)
#         perceptual_loss_h = torch.nn.functional.l1_loss(cH_SR, cH_HR)
#         perceptual_loss_v = torch.nn.functional.l1_loss(cV_SR, cV_HR)
#         perceptual_loss_d = torch.nn.functional.l1_loss(cD_SR, cD_HR)

#         return perceptual_loss_h + perceptual_loss_v + perceptual_loss_d
# class WTLoss(nn.Module):
#     def __init__(self):
#         super(WTLoss, self).__init__()

#     def wavelet_transform(self, image_tensor, wavelet='haar'):
#         # Assume that we will replace this with a PyTorch-compatible wavelet transform
#         batch_size, channels, height, width = image_tensor.shape
#         cA, cH, cV, cD = [], [], [], []
#         for i in range(batch_size):
#             image_np = image_tensor[i].permute(1, 2, 0).detach().cpu().numpy()  # to numpy array
#             coeffs = [pywt.dwt2(image_np[:, :, c], wavelet) for c in range(channels)]
#             cA_, (cH_, cV_, cD_) = map(np.dstack, zip(*coeffs))
#             cA.append(cA_)
#             cH.append(cH_)
#             cV.append(cV_)
#             cD.append(cD_)
#         return map(torch.tensor, (np.stack(cA), np.stack(cH), np.stack(cV), np.stack(cD)))

#     def forward(self, img_SR, img_HR):
#         _, cH, cV, cD = self.wavelet_transform(img_SR)
#         _, cH2, cV2, cD2 = self.wavelet_transform(img_HR)
#         cH, cV, cD = cH.cuda(), cV.cuda(), cD.cuda()
#         cH2, cV2, cD2 = cH2.cuda(), cV2.cuda(), cD2.cuda()
#         cH, cV, cD = cH.cuda(), cV.cuda(), cD.cuda()
#         # print(cH.shape)
#         new_tensor = torch.stack([cH, cV], dim=1)
#         # print(new_tensor.shape)
#         new_tensor2 = torch.stack([cH2, cV2], dim=1)
#         # print(new_tensor.shape,new_tensor2.size(0))
#         perceptual_loss=0
#         for channel1 in range(new_tensor2.size(0)):
#             channel_data = new_tensor[channel1,:, :, :]
#             channel_data2 = new_tensor2[channel1,:, :, :]
#             perceptual_loss += torch.nn.functional.l1_loss(channel_data, channel_data2.detach())
#         return perceptual_loss/new_tensor2.size(0)

    # def forward(self, img_SR, img_HR):
    #     _, cH, cV, cD = self.wavelet_transform(img_SR)
    #     _, cH2, cV2, cD2 = self.wavelet_transform(img_HR)

    #     cH, cV, cD = cH.cuda(), cV.cuda(), cD.cuda()
    #     cH2, cV2, cD2 = cH2.cuda(), cV2.cuda(), cD2.cuda()

    #     new_tensor = torch.stack([cH, cV, cD], dim=1)
    #     new_tensor2 = torch.stack([cH2, cV2, cD2], dim=1)
    #     print(new_tensor.shape,new_tensor2.size(0))
    #     perceptual_loss=0
    #     for channel1 in range(new_tensor2.size(0)):
    #         channel_data = new_tensor[channel1,:, :, :]
    #         channel_data2 = new_tensor2[channel1,:, :, :]
    #         perceptual_loss += torch.nn.functional.l1_loss(channel_data, channel_data2)
    #     return perceptual_loss


# class WTLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     #self 将它定义为一个实例方法
#     def wavelet_transform(self,image_tensor, wavelet='haar'):
#             # 获取批量大小
#             batch_size = image_tensor.size(0)
#             # 创建空列表来存储每个图像的小波变换结果
#             cA_list, cH_list, cV_list, cD_list = [], [], [], []
#             for i in range(batch_size):
#             # 将每个图像的 tensor 转换为 numpy 数组
#                 image_np = image_tensor[i].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
#                 # 分离颜色通道
#                 channels = [image_np[:, :, j] for j in range(3)]
#                 # 对每个通道进行小波变换
#                 coeffs = [pywt.dwt2(ch, wavelet) for ch in channels]
#                 # 重构子带图像
#                 cA, (cH, cV, cD) = map(np.dstack, zip(*coeffs))
#                 # 将小波变换结果添加到列表中
#                 cA_list.append(cA)
#                 cH_list.append(cH)
#                 cV_list.append(cV)
#                 cD_list.append(cD)
#             # 将列表转换为 numpy 数组
#             cA_array = np.stack(cA_list)
#             cH_array = np.stack(cH_list)
#             cV_array = np.stack(cV_list)
#             cD_array = np.stack(cD_list)
#             return cA_array, cH_array, cV_array, cD_array
#     def forward(self, img_SR, img_HR):
#         _, cH, cV, cD = self.wavelet_transform(img_SR)
#         _, cH2, cV2, cD2 = self.wavelet_transform(img_HR)
#         print(torch.tensor(cV).shape)
#         cH=torch.tensor(cH).cuda()
#         cV=torch.tensor(cV).cuda()
#         cD=torch.tensor(cD).cuda()
#         cH2=torch.tensor(cH2).cuda()
#         cV2=torch.tensor(cV2).cuda()
#         cD2=torch.tensor(cD2).cuda()
#         new_tensor=torch.randn(cH.size(0),3,cH.size(1),cH.size(2))
#         new_tensor2=torch.randn(cH.size(0),3,cH.size(1),cH.size(2))
#         for i in range(cH.size(0)):
#             slice_H = cH[i, :, :]
#             slice_V = cV[i, :, :]
#             slice_D = cD[i, :, :]
#             slice_H2 = cH2[i, :, :]
#             slice_V2 = cV2[i, :, :]
#             slice_D2 = cD2[i, :, :]
#             new_tensor[i] = torch.stack([slice_H, slice_V, slice_D])
#             new_tensor2[i] = torch.stack([slice_H2, slice_V2, slice_D2])
#         # combined_list =np.stack((cH, cV, cD), axis=0)
#         # combined_list2 = np.stack((cH2, cV2, cD2), axis=0)
#         # print("combined_list",torch.tensor(combined_list).shape)
#         # # 使用元组构造将列表转换为一个三维元组
#         # tensor_data = torch.tensor(combined_list,requires_grad=True).cuda()
#         # tensor_data2 = torch.tensor(combined_list2,requires_grad=True).cuda()
#         print(new_tensor.shape,new_tensor2.size(0))
#         perceptual_loss=0
#         for channel1 in range(new_tensor2.size(0)):
#             channel_data = new_tensor[channel1,:, :, :]
#             channel_data2 = new_tensor2[channel1,:, :, :]
#             perceptual_loss += torch.nn.functional.l1_loss(channel_data, channel_data2)
#         return perceptual_loss

# class WTLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#     #self 将它定义为一个实例方法
#     def wavelet_transform(self,image_tensor, wavelet='haar'):
#             # 获取批量大小
#             batch_size = image_tensor.size(0)

#             # 创建空列表来存储每个图像的小波变换结果
#             cA_list, cH_list, cV_list, cD_list = [], [], [], []
#             for i in range(batch_size):
#             # 将每个图像的 tensor 转换为 numpy 数组
#                 image_np = image_tensor[i].squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

#                 # 分离颜色通道
#                 channels = [image_np[:, :, j] for j in range(3)]

#                 # 对每个通道进行小波变换
#                 coeffs = [pywt.dwt2(ch, wavelet) for ch in channels]

#                 # 重构子带图像
#                 cA, (cH, cV, cD) = map(np.dstack, zip(*coeffs))

#                 # 将小波变换结果添加到列表中
#                 cA_list.append(cA)
#                 cH_list.append(cH)
#                 cV_list.append(cV)
#                 cD_list.append(cD)

#             # 将列表转换为 numpy 数组
#             cA_array = np.stack(cA_list)
#             cH_array = np.stack(cH_list)
#             cV_array = np.stack(cV_list)
#             cD_array = np.stack(cD_list)

#             return cA_array, cH_array, cV_array, cD_array
#     def forward(self, img_SR, img_HR):
#         _, cH, cV, cD = self.wavelet_transform(img_SR)
#         _, cH2, cV2, cD2 = self.wavelet_transform(img_HR)
#         combined_list =np.stack((cH, cV, cD), axis=0)
#         combined_list2 = np.stack((cH2, cV2, cD2), axis=0)
#         # 使用元组构造将列表转换为一个三维元组
#         tensor_data = torch.tensor(combined_list,requires_grad=True).cuda()
#         tensor_data2 = torch.tensor(combined_list2,requires_grad=True).cuda()
#         perceptual_loss=0
#         for channel1 in range(tensor_data.size(0)):
#             channel_data = tensor_data[channel1,:, :, :]
#             channel_data2 = tensor_data2[channel1,:, :, :]
#             perceptual_loss += torch.nn.functional.l1_loss(channel_data, channel_data2)
#         return perceptual_loss
# class WTLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def wavelet_transform(self, image_tensor, wavelet='haar'):
#         batch_size = image_tensor.size(0)
#         cA, cH, cV, cD = [], [], [], []
#         for i in range(batch_size):
#             image_np = image_tensor[i].permute(1, 2, 0).cpu().numpy()
#             coeffs = [pywt.dwt2(image_np[:, :, j], wavelet) for j in range(3)]
#             cA.append(np.dstack([c[0] for c in coeffs]))
#             cH.append(np.dstack([c[1][0] for c in coeffs]))
#             cV.append(np.dstack([c[1][1] for c in coeffs]))
#             cD.append(np.dstack([c[1][2] for c in coeffs]))
#         return map(lambda x: torch.tensor(np.stack(x)).permute(0, 3, 1, 2).cuda(), (cA, cH, cV, cD))

#     def forward(self, img_SR, img_HR):
#         _, cH, cV, cD = self.wavelet_transform(img_SR)
#         _, cH2, cV2, cD2 = self.wavelet_transform(img_HR)

#         perceptual_loss = torch.nn.functional.l1_loss(torch.stack([cH, cV, cD], dim=1), 
#                                                       torch.stack([cH2, cV2, cD2], dim=1))
#         return perceptual_loss

# class PerceptualLoss():
#     def __init__(self, loss):
#         self.criterion = loss
#         self.contentFunc = self.contentFunc()
#     def contentFunc(self):
#         conv_3_3_layer =14 
#         cnn = models.vgg19(pretrained=True).features
#         cnn = cnn.cuda()
#         model = nn.Sequential()
#         model = model.cuda()
#         for i, layer in enumerate(list(cnn)):
#             model.add_module(str(i), layer)
#             if i == conv_3_3_layer:
#                 break
#         for param in model.parameters():
#             param.requires_grad = False
#         return model
#     def get_loss(self, fakeIm, realIm):
#         f_fake = self.contentFunc.forward(fakeIm)
#         f_real = self.contentFunc.forward(realIm)
#         f_real_no_grad = f_real.detach()
#         loss = self.criterion(f_fake, f_real_no_grad)
#         return loss
#     def __call__(self, input, target):
#         # 调用 get_loss 方法计算损失
#         return self.get_loss(input, target)
class WTLoss(nn.Module):
    def __init__(self):
        super(WTLoss, self).__init__()
    def forward(self, x, y, J=4):
        # Perform 4-level 2D discrete wavelet transform on both images
        x_dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
        y_dwt_f = pw.DWTForward(J=J, wave='haar', mode='symmetric')
        x_dwt_f.cuda()
        y_dwt_f.cuda()
        x_dwt = x_dwt_f(x)[1]
        y_dwt = y_dwt_f(y)[1]
        h_mse, v_mse, d_mse = 0, 0, 0
        for i in range(J):
            # Calculate MSE between the coefficients of each subband
            h_mse += torch.nn.functional.mse_loss(x_dwt[i][:, :, 0, :, :], y_dwt[i][:, :, 0, :, :])
            v_mse += torch.nn.functional.mse_loss(x_dwt[i][:, :, 1, :, :], y_dwt[i][:, :, 1, :, :])
            d_mse += torch.nn.functional.mse_loss(x_dwt[i][:, :, 2, :, :], y_dwt[i][:, :, 2, :, :])

        # Sum the MSE losses across subbands and return
        return h_mse + v_mse + d_mse
class PerceptualLoss():
    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def contentFunc(self):
        feature_layers = [0, 3, 8, 17, 26, 35]
        cnn = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()

        # 添加一个变量来记录当前要添加到model中的层的索引
        current_layer = 0
        for i, layer in enumerate(list(cnn)):
            if current_layer < len(feature_layers) and i <= feature_layers[current_layer]:
                model.add_module(str(i), layer)
                if i == feature_layers[current_layer]:
                    current_layer += 1
            if current_layer >= len(feature_layers):  # 如果已添加所有指定层，则停止循环
                break

        for param in model.parameters():
            param.requires_grad = False
        # print(model)
        return model

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss
    def __call__(self, input, target):
        return self.get_loss(input, target)


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)

