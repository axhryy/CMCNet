import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_wavelets as pw
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
