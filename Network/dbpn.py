import torch
import torch.nn as nn
from base_networks import ConvBlock, UpBlock, DownBlock, D_DownBlock, D_UpBlock

'''
    DBPN模块是深度反向投影网络（Deep Back-Projection Network）的核心组成部分，它通过借鉴传统方法中的反向投影（Back Projection）方法，构造了迭代式升降采样（Iterative up and downsampling）的方法，实现了超出以往的超分辨率  （Super Resolution）效果。DBPN模块主要有两种类型：up-projection和down-projection，分别用于将低分辨率特征映射到高分辨率特征，或者将高分辨率特征映射到低分辨率特征。DBPN模块可以有效地利用低分辨率图像和高分辨率图像之间共有的关系，提高图像质量。
'''


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        #Back-projection stages
        self.up1 = UpBlock(base_filter, kernel, stride, padding)  #1c  2de
        self.down1 = DownBlock(base_filter, kernel, stride, padding)  #2c 1de
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)  #3c  1de
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)   #2c 2de
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        #Reconstruction
        self.output_conv = ConvBlock(num_stages*base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, x):
        x = self.feat0(x)
        x = self.feat1(x)
        
        h1 = self.up1(x)
        l1 = self.down1(h1)
        h2 = self.up2(l1)
        
        concat_h = torch.cat((h2, h1), 1)
        lr = self.down2(concat_h)
        
        concat_l = torch.cat((lr, l1), 1)
        h = self.up3(concat_l)
        
        concat_h = torch.cat((h, concat_h), 1)
        lr = self.down3(concat_h)
        
        concat_l = torch.cat((lr, concat_l), 1)
        h = self.up4(concat_l)
        
        concat_h = torch.cat((h, concat_h), 1)
        lr = self.down4(concat_h)
        
        concat_l = torch.cat((lr, concat_l), 1)
        h = self.up5(concat_l)
        
        concat_h = torch.cat((h, concat_h), 1)
        lr = self.down5(concat_h)
        
        concat_l = torch.cat((lr, concat_l), 1)
        h = self.up6(concat_l)
        
        concat_h = torch.cat((h, concat_h), 1)
        lr = self.down6(concat_h)
        
        concat_l = torch.cat((lr, concat_l), 1)
        h = self.up7(concat_l)
        
        concat_h = torch.cat((h, concat_h), 1)
        x = self.output_conv(concat_h)
        
        return x
