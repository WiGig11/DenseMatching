import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List
import torchvision.models as models
from torchvision.models.swin_transformer import Swin_B_Weights
from collections import OrderedDict

class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class SwinPyramid(nn.Module):
    """
    # *! Variants1 :
    Directly used the swin transformer as the backbone and the patch merge as the downsampler
    """
    def __init__(self, train=False, pretrained=True):
        super().__init__()
        self.swin = models.swin_transformer.swin_b(weights=Swin_B_Weights.DEFAULT)
        for param in self.swin.parameters():
            param.requires_grad = train

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        feat1 = self.swin.features[1](self.swin.features[0](x))
        feat2 = self.swin.features[3](self.swin.features[2](feat1))
        feat3 = self.swin.features[5](self.swin.features[4](feat2))
        feat4 = self.swin.features[7](self.swin.features[6](feat3))
        outputs = []

        if quarter_resolution_only:
            x_quarter = feat1
            outputs.append(x_quarter.permute(0, 3, 1, 2))
        elif eigth_resolution:
            x_quarter = feat1
            outputs.append(x_quarter.permute(0, 3, 1, 2))
            x_eight = feat2
            outputs.append(x_eight.permute(0, 3, 1, 2))
        else:
            outputs.append(feat1.permute(0, 3, 1, 2))
            outputs.append(feat2.permute(0, 3, 1, 2))
            outputs.append(feat3.permute(0, 3, 1, 2))
            outputs.append(feat4.permute(0, 3, 1, 2))
        return outputs
    
class SwinVGGPyramid(nn.Module):
    """
    # *!Variant 2
    #  Directly add long range reliance on the original VGG pyrimaid
    """
    def __init__(self, train=False, pretrained=True):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=pretrained)
        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False
        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules
        self.swin = models.swin_transformer.swin_b(weights=Swin_B_Weights.DEFAULT)
        for param in self.swin.parameters():
            param.requires_grad = train
        
        self.patch_embedding1 = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=(4,4), stride=(4,4)),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm((512), eps=1e-05, elementwise_affine=True),
        )
        self.patch_embedding2 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=(4,4), stride=(4,4)),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm((512), eps=1e-05, elementwise_affine=True),
        )
        self.patch_embedding3 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=(4,4), stride=(4,4)),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm((512), eps=1e-05, elementwise_affine=True),
        )
        self.patch_embedding4 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=(4,4), stride=(4,4)),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm((512), eps=1e-05, elementwise_affine=True),
        )
        self.channel_dropper1 = nn.ConvTranspose2d(512,3,kernel_size = (4,4),stride = (4,4))
        self.channel_dropper2 = nn.ConvTranspose2d(512,64,kernel_size = (4,4),stride = (4,4))
        self.channel_dropper3 = nn.ConvTranspose2d(512,64,kernel_size = (4,4),stride = (4,4))
        self.channel_dropper4_520 = nn.ConvTranspose2d(512,128,kernel_size = (6,6),stride = (4,4))
        self.channel_dropper4_256 = nn.ConvTranspose2d(512,128,kernel_size = (4,4),stride = (4,4))


    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        # TODO 在大尺度上进行手写的pe，转为token，然后通过堆叠的swin，得到结果Tensor，然后把permute的结果放入结果中，swin后的结果应该是高维512的，因此需要1*1卷积降维
        # *! 长距离依赖应该是在更大尺度上有意义！
        outputs = []
        if quarter_resolution_only:
            x = self.channel_dropper1(self.swin.features[5](self.patch_embedding1(x)).permute(0,3,1,2))
            x_full = self.__dict__['_modules']['level_' + str(0)](x)

            x_full = self.channel_dropper2(self.swin.features[5](self.patch_embedding2(x_full)).permute(0,3,1,2))
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x = self.channel_dropper1(self.swin.features[5](self.patch_embedding1(x)).permute(0,3,1,2))
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            
            x_full = self.channel_dropper2(self.swin.features[5](self.patch_embedding2(x_full)).permute(0,3,1,2))
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_half = self.channel_dropper3(self.swin.features[5](self.patch_embedding3(x_half)).permute(0,3,1,2))
            
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            
            if x_quarter.shape[2]==130:
                x_quarter = self.channel_dropper4_520(self.swin.features[5](self.patch_embedding4(x_quarter)).permute(0,3,1,2))
            else:
                x_quarter = self.channel_dropper4_256(self.swin.features[5](self.patch_embedding4(x_quarter)).permute(0,3,1,2))
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
            
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)
        return outputs
    
class SwinOnceVGGPyramid(nn.Module):
    """
    # *!Variant 3
    #  Swin only once on the full reso
    """
    def __init__(self, train=False, pretrained=True):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=pretrained)
        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False
        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules
        self.swin = models.swin_transformer.swin_b(weights=Swin_B_Weights.DEFAULT)
        for param in self.swin.parameters():
            param.requires_grad = train
        
        self.patch_embedding = self.swin.features[0]
        self.channel_dropper = nn.ConvTranspose2d(128,3,kernel_size = (4,4),stride = (4,4))

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        x = self.channel_dropper(self.swin.features[1](self.patch_embedding(x)).permute(0,3,1,2))
        outputs = []
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)
        return outputs



'''
#*! 调试用

inputtensor = torch.randn(1,3,520,520)
svp = SwinOnceVGGPyramid()

feat = svp.forward(inputtensor,eigth_resolution = True)
#print(feat[-2].shape)
#print(feat[-1].shape)

Vgg = VGGPyramid()
feat = Vgg(inputtensor)
feat = Vgg.forward(inputtensor,eigth_resolution = True)
svp = SwinPyramid()
inputtensor = torch.randn(1,3,256,256)
feat = svp.forward(inputtensor,eigth_resolution = True)
print(feat[-2].shape)
print(feat[-1].shape)
Vgg = VGGPyramid()
feat = Vgg(inputtensor)
feat = Vgg.forward(inputtensor,eigth_resolution = True)
print(feat[0].shape)
print(feat[1].shape)
print(feat[2].shape)


#*? 调试用?

inputtensor = torch.randn(1,3,520,520)
vit = SwinVGGPyramid()
output = vit(inputtensor)
print(output.shape)
class VGGPyramid(nn.Module):
    def __init__(self, train=False, pretrained=True):
        super().__init__()
        self.n_levels = 5
        source_model = models.vgg16(pretrained=pretrained)

        modules = OrderedDict()
        tmp = []
        n_block = 0
        first_relu = False

        for c in source_model.features.children():
            if (isinstance(c, nn.ReLU) and not first_relu) or (isinstance(c, nn.MaxPool2d)):
                first_relu = True
                tmp.append(c)
                modules['level_' + str(n_block)] = nn.Sequential(*tmp)
                for param in modules['level_' + str(n_block)].parameters():
                    param.requires_grad = train

                tmp = []
                n_block += 1
            else:
                tmp.append(c)

            if n_block == self.n_levels:
                break

        self.__dict__['_modules'] = modules

    def forward(self, x, quarter_resolution_only=False, eigth_resolution=False):
        outputs = []
        if quarter_resolution_only:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
        elif eigth_resolution:
            x_full = self.__dict__['_modules']['level_' + str(0)](x)
            outputs.append(x_full)
            x_half = self.__dict__['_modules']['level_' + str(1)](x_full)
            x_quarter = self.__dict__['_modules']['level_' + str(2)](x_half)
            outputs.append(x_quarter)
            x_eight = self.__dict__['_modules']['level_' + str(3)](x_quarter)
            outputs.append(x_eight)
        else:
            for layer_n in range(0, self.n_levels):
                x = self.__dict__['_modules']['level_' + str(layer_n)](x)
                outputs.append(x)
        return outputs

#self.transformer = models.vision_transformer.vit_b_16(pretrained=pretrained)
swin = models.swin_transformer.swin_b(weights=Swin_B_Weights.DEFAULT)
print(swin)
#print(swin.features[0])

inputtensor = torch.randn(1,3,520,520)
feat = swin.features[1](swin.features[0](inputtensor))
print(feat.shape)
feat = swin.features[3](swin.features[2](feat))
print(feat.shape)
feat = swin.features[5](swin.features[4](feat))
print(feat.shape)
feat = swin.features[7](swin.features[6](feat))
print(feat.shape)
feat = swin.norm(feat)
print(feat.shape)
feat = swin.permute(feat)
print(feat.shape)
feat = swin.avgpool(feat)
print(feat.shape)
feat = swin.flatten(feat)
print(feat.shape)
feat = swin.head(feat)
print(feat.shape)

print('---------------===============================')

inputtensor = torch.randn(1,3,256,256)
feat = swin.features[1](swin.features[0](inputtensor))
print(feat.shape)
feat = swin.features[3](swin.features[2](feat))
print(feat.shape)
feat = swin.features[5](swin.features[4](feat))
print(feat.shape)
feat = swin.features[7](swin.features[6](feat))
print(feat.shape)

print('---------------===============================')
Vgg = VGGPyramid()
inputtensor = torch.randn(1,3,520,520)
feat = Vgg.forward(inputtensor,eigth_resolution = True)
print(feat[0].shape)
print(feat[1].shape)
print(feat[2].shape)



print('---------------===============================')

inputtensor = torch.randn(1,3,256,256)
Vgg = VGGPyramid()
feat = Vgg(inputtensor)
feat = Vgg.forward(inputtensor,eigth_resolution = True)
print(feat[0].shape)
print(feat[1].shape)
print(feat[2].shape)

svp = SwinVGGPyramid()
inputtensor = torch.randn(1,3,256,256)
feat = svp.forward(inputtensor,eigth_resolution = True)
print(feat[-2].shape)
print(feat[-1].shape)

#inputtensor = torch.randn(1,3,256,256)
seq = swin.features[0]
blk1 = swin.features[1]
merge = swin.features[2]
print('---------------===============================')
print(seq)
output = seq(inputtensor)
print(output.shape)

print('---------------===============================')
print(blk1)
output = blk1(output)
print(output.shape)

print('---------------===============================')
print(merge)
output = merge(output)
print(output.shape)
'''
