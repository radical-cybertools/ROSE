import torch.nn as nn
from src.layers.layers import MLPBlock, ConvBlock
from src.models.NetworkStructureSampler import NetworkStructureSampler
import numpy as np
import torch
import torch.nn.functional as F
import math


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        sh = (x.shape[0],) + self.shape
        return x.view(sh)


class SimpleClas(nn.Module):
    def __init__(self):
        super(SimpleClas, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.Dropout2d(0.4),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Dropout2d(0.4),
                                 nn.BatchNorm2d(1),
                                )
        # self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
        #                            nn.LeakyReLU(negative_slope=0.02),
        #                            nn.BatchNorm2d(64),
        #                            nn.Dropout2d(0.1),
        #                            nn.AvgPool2d(2),

        #                            nn.Conv2d(64, 128, 3, 1, 1),
        #                            nn.LeakyReLU(negative_slope=0.02),
        #                            nn.AvgPool2d(2),

        #                            nn.Conv2d(128, 256, 3, 1, 1),
        #                            nn.LeakyReLU(negative_slope=0.02),
        #                            nn.AvgPool2d(2),

        #                            nn.Conv2d(256, 512, 3, 1, 1),
        #                            nn.LeakyReLU(negative_slope=0.02),
        #                            nn.AvgPool2d(2),

        #                            nn.Conv2d(512, 1024, 3, 1, 1),
        #                            nn.LeakyReLU(negative_slope=0.02),
        #                            nn.AvgPool2d(2),

        #                            nn.Conv2d(1024, 2048, 3, 1, 0),
        #                            nn.LeakyReLU(negative_slope=0.02),
        #                            # nn.AvgPool2d(2), # USE for IMAGES 128x128
        #                            nn.Conv2d(2048, 4096, 2, 1, 0),  # USE for IMAGES 128x128

        #                            nn.Flatten(),

        #                            nn.Linear(4096, 512),
        #                            nn.Linear(512, 10),
        #                            nn.LogSoftmax(dim=1))  #

    def forward(self, x):
        x = self.seqIn(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   # nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x):
        x1 = self.seqIn(x)
        x1 = self.seqOut(x1)
        return x1
    
class SimpleCNN100(nn.Module):
    def __init__(self): #Same as previous but specifically for 100x100 input
        super(SimpleCNN100, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                #    nn.Dropout2d(0.4),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                #    nn.Dropout2d(0.4),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #6

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),  #12

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #24

#                                     nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.ConvTranspose2d(128,64,2,1,0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2))

        self.out_layer = nn.Sequential(nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02))

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x1 = self.seqIn(x)
                e = self.seqOut(x1)
            x1 = self.out_layer(e)
        else:
            x1 = self.seqIn(x)
            e = self.seqOut(x1)
            x1 = self.out_layer(e)
        if last:
            return e, x1
        return x1
    
class JuliaCNN100(nn.Module):
    def __init__(self, dout1=0.4, dout2=0.4, dout3=0.1, dout4=0.1, p1=0.5):
        super(JuliaCNN100, self).__init__() #pytorch version of the Julia network
        self.dout1 = dout1
        self.dout2 = dout2
        self.dout3 = dout3
        self.dout4 = dout4
        self.p1 = p1
        self.p2 = 1.0 - p1
        self.nn1 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.Dropout2d(self.dout1),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Dropout2d(self.dout2),
                                 nn.BatchNorm2d(1),
                                )
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(self.dout3),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 2, 0),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 2, 0),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2), #6

                                    nn.ConvTranspose2d(512, 256, 3, 2, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),  #12

                                    nn.ConvTranspose2d(256, 128, 3, 2, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2), #24

#                                     nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.ConvTranspose2d(128,64,3,2,0),
                                    nn.LeakyReLU(negative_slope=0.02),
#                                     nn.Upsample(scale_factor=2),

                                    nn.Dropout2d(self.dout4),
                                    nn.ConvTranspose2d(64, 1, 4, 2, 2),
                                    nn.ReLU(),
#                                     nn.BatchNorm2d(1),

                                   )
        self.nn2 = nn.Sequential(self.seqIn,
                                 self.seqOut,        
                                )
        
    def updatePs(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        xout = self.p1 * x1 + self.p2 * x2
        return xout
    
class JuliaCNN100_2(nn.Module):
    def __init__(self, dout1=0.4, dout2=0.4, dout3=0.1, dout4=0.1, p1=0.5):
        super(JuliaCNN100_2, self).__init__() #pytorch version of the Julia network
        self.dout1 = dout1
        self.dout2 = dout2
        self.dout3 = dout3
        self.dout4 = dout4
        self.p1 = p1
        self.p2 = 1.0 - p1
        self.nn1 = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.Dropout2d(self.dout1),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.LeakyReLU(negative_slope=0.02),
                                 nn.BatchNorm2d(1),
                                 
                                 nn.Conv2d(1, 1, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Dropout2d(self.dout2),
                                 nn.BatchNorm2d(1),
                                )
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
#                                    nn.AvgPool2d(2),
                                   )

        self.seqOut = nn.Sequential(nn.ConvTranspose2d(2048, 1024, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),

                                    nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #6

                                    nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),  #12

                                    nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2), #24

#                                     nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                    nn.ConvTranspose2d(128,64,2,1,0),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(64, 32, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02),
                                    nn.Upsample(scale_factor=2),

                                    nn.ConvTranspose2d(32, 1, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.02)
                                   )
        
        self.nn2 = nn.Sequential(self.seqIn,
                                 self.seqOut,        
                                )
        
    def updatePs(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def forward(self, x):
        x1 = self.nn1(x)
        x2 = self.nn2(x)
        xout = self.p1 * x1 + self.p2 * x2
        return xout


class DNN(nn.Module):
    def __init__(self, p1=1.0, p2=0.0, dout1=0.1, dout2=0.1):
        super(DNN, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.dout1 = dout1
        self.dout2 = dout2
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Dropout2d(self.dout1),
                                   nn.BatchNorm2d(64),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Conv2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(4),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(1024, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(512, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(256, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.ConvTranspose2d(128, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.LeakyReLU(negative_slope=0.02),

                                   nn.Dropout2d(self.dout2),
                                   nn.ConvTranspose2d(64, 1, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.0),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   # nn.ReLU(),
                                   # nn.BatchNorm2d(1),

                                   )

    def forward(self, x):
        x = self.p1 * self.seqIn(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.blk1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blk5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.upConv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(1024, 512, 3, 1, 1),
        )
        self.upConv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
        )
        self.upConv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
        )
        self.upConv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
        )
        self.upConvLast = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x1 = self.blk1(x)
        x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1))
        x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2))
        x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3))
        x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4))

        x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
        x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
        x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
        x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
        # print(x9.shape, x9.is_cuda)
        xfinal = self.upConvLast(x9)
        # print(xfinal.shape)

        return xfinal

class AbsorbingConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AbsorbingConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Pad input tensor with periodic boundary conditions
        # x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        x_pad_neg = F.pad(x, [self.padding, self.padding, self.padding, self.padding], mode='reflect')
        x_pad_neg = torch.neg(x_pad_neg)
        x_pad_neg[:,:,self.padding:-self.padding,self.padding:-self.padding] = x
        # Apply convolution
        x = self.conv(x_pad_neg)
        return x
        
class UNetAB(nn.Module):
    def __init__(self):
        super(UNetAB, self).__init__()
        self.blk1 = nn.Sequential(
            AbsorbingConv2d(1, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
            AbsorbingConv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
        )
        self.blk2 = nn.Sequential(
            AbsorbingConv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
            AbsorbingConv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
        )
        self.blk3 = nn.Sequential(
            AbsorbingConv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
            AbsorbingConv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
        )
        self.blk4 = nn.Sequential(
            AbsorbingConv2d(256, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
            AbsorbingConv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
        )
        self.blk5 = nn.Sequential(
            AbsorbingConv2d(512, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
            AbsorbingConv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
            # nn.Dropout2d(0.4),
        )
        
        self.blkUp1 = nn.Sequential(
            AbsorbingConv2d(1024, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            AbsorbingConv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp2 = nn.Sequential(
            AbsorbingConv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            AbsorbingConv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp3 = nn.Sequential(
            AbsorbingConv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            AbsorbingConv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.blkUp4 = nn.Sequential(
            AbsorbingConv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),

            AbsorbingConv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.0),
        )
        self.upConv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(1024, 512, 3, 1, 1),
        )
#         self.upConv2 = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ConvTranspose2d(512, 256, 3, 1, 1),
#         )
        self.upConv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 0),
        )
        self.upConv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
        )
        self.upConv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
        )
        self.lastlayer = nn.ConvTranspose2d(64, 1, 3, 1, 1)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                x1 = self.blk1(x) #512
                x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
                x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
                x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
                x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32

                x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
                x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
                x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
                x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
            xfinal = self.lastlayer(x9)
        else:
            x1 = self.blk1(x) #512
            # print(x1.shape)
            x2 = self.blk2(nn.MaxPool2d(2, stride=2)(x1)) #256
            # print(x2.shape)
            x3 = self.blk3(nn.MaxPool2d(2, stride=2)(x2)) #128
            # print(x3.shape)
            x4 = self.blk4(nn.MaxPool2d(2, stride=2)(x3)) #64
            # print(x4.shape)
            x5 = self.blk5(nn.MaxPool2d(2, stride=2)(x4)) #32
            # print(x5.shape)

            x6 = self.blkUp1(torch.cat((self.upConv1(x5), x4), dim=1))
            # print(x6.shape)
            x7 = self.blkUp2(torch.cat((self.upConv2(x6), x3), dim=1))
            # print(x7.shape)
            x8 = self.blkUp3(torch.cat((self.upConv3(x7), x2), dim=1))
            # print(x8.shape)
            x9 = self.blkUp4(torch.cat((self.upConv4(x8), x1), dim=1))
            # print(x9.shape)
            xfinal = self.lastlayer(x9)
            # print(xfinal.shape)
            # exit(0)
        if last:
            return x9, xfinal
        return xfinal

class SimpleDisc(nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.seqIn = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.BatchNorm2d(64),
                                   nn.Dropout2d(0.1),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(64, 128, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(128, 256, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(256, 512, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(512, 1024, 3, 1, 1),
                                   nn.LeakyReLU(negative_slope=0.02),
                                   nn.AvgPool2d(2),

                                   nn.Conv2d(1024, 2048, 3, 1, 0),
                                   nn.LeakyReLU(negative_slope=0.02),

                                   nn.Flatten(),

                                   nn.Linear(2048, 512),
                                   nn.Linear(512, 1),
                                   nn.Sigmoid())  #

    def forward(self, x):
        x = self.seqIn(x)
        return x


class SimpleGen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.seqIn = nn.Sequential(nn.Linear(100, 160000),
                                   nn.BatchNorm1d(160000),
                                   nn.ReLU(),
                                   Reshape(256, 25, 25),
                                   nn.ConvTranspose2d(256, 128, 5, 2, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(128, 64, 4, 1, 2),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64, 1, 4, 2, 1),
                                   nn.Hardtanh()
                                   )  #

    def forward(self, x):
        x = self.seqIn(x)
        return x

class AdaptiveConvNet(nn.Module):
    def __init__(self, input_channels, num_classes, num_channels=1, kernel_size=3, args=None, device=None):
        super(AdaptiveConvNet, self).__init__()
        self.mode = "NN"
        self.args = args
        print(args)

        self.max_channels = num_channels
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.truncation_level = args.truncation_level
        self.num_samples = args.num_samples
        self.device = device

        self.structure_sampler = NetworkStructureSampler(args, self.device)

        self.layers = nn.ModuleList([ConvBlock(self.input_channels, self.max_channels,
                                      kernel_size=3, padding=1, stride=1, pool=False, dropout=True, act="leaky").to(self.device)])

        for i in range(1, self.truncation_level):
            self.layers.append(ConvBlock(self.max_channels,
                                         self.max_channels,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                        residual=True).to(self.device))

        self.out_layer = ConvBlock(1, 1, kernel_size=3, padding=1, stride=1, dropout=True, act=None)


    def _forward(self, x, mask_matrix, threshold):
        """
        Transform the input feature matrix with a sampled network structure
        Parameters
        ----------
        x : input feature matrix
        mask_matrix : mask matrix from Beta-Bernoulli Process
        threshold : number of layers in sampled structure

        Returns
        -------
        out : Output from the sampled architecture
        """
        # if number of layers sampled during testing is greater than truncation level, we use the network with
        # truncation level.
        if not self.training and threshold > len(self.layers):
            threshold = len(self.layers)

        for layer in range(threshold):
            mask = mask_matrix[:, layer]
            x = self.layers[layer](x, mask)
        # return x

        out = self.out_layer(x)
        return out

    def forward(self, x, num_samples=5):
        """
        Transforms the input feature matrix with different samples of network structures Z

        Parameters
        ----------
        x : data
        num_samples : Number of samples of network structure Z

        Returns
        -------
        act_vec : Tensor by stacking the output from different structures Z
        """
        act_vec = []
        Z, threshold = self.structure_sampler(num_samples)

        for s in range(num_samples):
            out = self._forward(x, Z[s], threshold)
            act_vec.append(out.unsqueeze(0))
            # act_vec.append(out)

        act_vec = torch.cat(act_vec, dim=0)
        return act_vec

    def get_E_loglike(self, neg_loglike_fun, output, target):
        """

        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        output : scores predicted by model
        target : real labels

        Returns
        -------
        mean negative log likelihood of the model based on different architectures
        """

        num_samples = self.num_samples
        print(output.shape, target.shape, num_samples)
        batch_sze = target.shape[0]
        target_expand = target.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)
        print(output.shape, target.shape, target_expand.shape, num_samples)
        # output = output.view(num_samples * batch_sze, -1)
        # print(output.shape, target_expand.shape)
        neg_loglike = neg_loglike_fun(output, target_expand)#.view(num_samples, batch_sze)
        print(neg_loglike.shape); exit(0)
        E_neg_loglike = neg_loglike.mean(0).mean()
        return E_neg_loglike

    def estimate_ELBO(self, neg_loglike_fun, act_vec, y, N_train, kl_weight=1):
        """
        Estimate the ELBO
        Parameters
        ----------
        neg_loglike_fun : Negative log likelihood function
        act_vec : scores from different samples of latent neural architecture Z
        y : real labels
        N_train: number of training data points
        kl_weight: coefficient to scale KL component

        Returns
        -------
        ELBO
        """
        # E_loglike = self.get_E_loglike(neg_loglike_fun, act_vec, y)
        E_loglike = neg_loglike_fun(act_vec.mean(0), y)
        KL = self.structure_sampler.get_kl()
        ELBO = E_loglike + (kl_weight * KL)/N_train
        return ELBO
    

# class AdaptiveConvNet(nn.Module):
#     def __init__(self, input_channels, num_classes, num_channels=64, kernel_size=3, args=None, device=None):
#         super(AdaptiveConvNet, self).__init__()
#         self.mode = "NN"
#         self.args = args

#         self.num_channels = num_channels
#         self.kernel_size = kernel_size
#         self.input_channels = input_channels
#         self.num_classes = num_classes

#         # self.task = args.task
#         self.truncation = args.truncation
#         self.num_samples = args.num_samples
#         self.device = device

#         if self.args.increasing_filters:
#             layer_size = {16: 0.3, 32: 0.3, 64: 0.4}
#             self.layer_sizes = np.concatenate([[k] * int(v * self.args.truncation) for k, v in layer_size.items() if
#                                                int(v * self.args.truncation) > 0])
#             self.out_layer_size = 1024
#         else:
#             layer_size = [num_channels] * self.args.truncation
#             self.layer_sizes = np.array(layer_size)
#             self.out_layer_size = num_channels
#             # print(num_channels, self.args.truncation, self.layer_sizes, self.out_layer_size)

#         self.architecture_sampler = SampleNetworkArchitecture(args, self.device)

#         self.layers = nn.ModuleList([ConvBlock(self.input_channels, self.layer_sizes[0],
#                                       kernel_size=3, padding=1, stride=1, pool=True).to(self.device)])

#         for i in range(1, self.args.truncation):
#             self.layers.append(ConvBlock(self.layer_sizes[i - 1],
#                                           self.layer_sizes[i], kernel_size=3, padding=1,
#                                           residual=True).to(self.device))
            
        
#         self.out_layer = nn.Sequential(MLPBlock(self.out_layer_size, self.num_channels, residual=True),
#                                        nn.Linear(self.num_channels, self.num_classes))


#     def set_optimizer(self, optimizer):
#         self.optimizer = optimizer

#     def add_param_to_optimizer(self, layer):
#         n_layers = len(self.layers)
#         for child in layer.children():
#             if isinstance(child, (nn.Conv2d, nn.Conv1d)):
#                 for name, param in child.named_parameters():
#                     if "weight" in name:
#                         self.register_parameter(f"conv{n_layers + 1}_weights", param)
#                     elif "bias" in name:
#                         self.register_parameter(f"conv{n_layers + 1}_bias", param)

#                     self.optimizer.add_param_group({"params": param})

#             elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
#                 for name, param in child.named_parameters():
#                     if "weight" in name:
#                         self.register_parameter(f"bn{n_layers + 1}_weights", param)
#                     elif "bias" in name:
#                         self.register_parameter(f"bns{n_layers + 1}_bias", param)

#                     self.optimizer.add_param_group({"params": param})

#     def update_model(self, new_threshold):
#         num_layers_available = len(self.layers)
#         if new_threshold > num_layers_available:
#             print("hjere")
#             flag = 1
#             num_new_layers = new_threshold - num_layers_available
#             for i in range(num_new_layers):
#                 if num_layers_available == 0:
#                     layer = ConvBlock(self.input_channels, self.layer_sizes[self.num_layers],
#                                       kernel_size=3, padding=1, pool=True).to(self.device)

#                     num_layers_available += 1
#                 else:
#                     layer = ConvBlock(self.layer_sizes[self.num_layers - 1],
#                                           self.layer_sizes[self.num_layers], kernel_size=3, padding=1,
#                                           residual=True).to(self.device)

#                 self.add_param_to_optimizer(layer)
#                 self.num_layers += 1
#                 self.layers.append(layer)
#                 layer = None

#     def forward(self, x, mask_matrix, threshold):
#         # print(x.shape)
#         if self.training:
#             self.update_model(threshold)

#         if not self.training and threshold > len(self.layers):
#             threshold = len(self.layers)
            
#         for k in range(threshold):
#             if isinstance(mask_matrix, list):
#                 mask = mask_matrix[k]
#             else:
#                 mask = mask_matrix[:, k]
#             x = self.layers[k](x, mask)

#         if self.args.increasing_filters:
#             pool_size = int(self.out_layer_size / x.shape[1])
#             if pool_size == 4 or pool_size == 16:
#                 p = int(math.sqrt(pool_size))
#                 x = F.adaptive_avg_pool2d(x, (p, p))
#             else:
#                 x = F.adaptive_avg_pool2d(x, (2, pool_size // 2))
#             x = x.view(x.shape[0], -1)
#         # else:
#         #     # print("Be", x.shape)
#         #     x = x.mean(dim=(2, 3))

#         return x, x
    
    

#     def vanilla_forward(self, x):

#         for layer in self.layers:
#             x = layer(x, 1)

#         out = self.out_layer(x)
#         return out


#     def get_embedding(self, x, num_samples):
#         kl = torch.Tensor([0.]).to(self.device)
#         Z, n_layers, thresholds = self.architecture_sampler(num_samples)
        
#         # print(Z, Z.shape)
#         # print(num_samples, thresholds); exit(0)
#         embedding = torch.tensor([])
#         for i in range(num_samples):
#             _, emb = self.forward(x, Z[i], n_layers)
#             emb = emb.unsqueeze(0)
#             if i == 0:
#                 embedding = emb
#             else:
#                 embedding = torch.cat((embedding, emb), dim=0)
#         embedding = torch.mean(embedding, 0)
#         return embedding
    
#     def get_output_embedding(self, x, num_samples):
#         kl = torch.Tensor([0.]).to(self.device)
#         Z, n_layers, thresholds = self.architecture_sampler(num_samples)
        
#         # print(Z, Z.shape)
#         # print(num_samples, thresholds); exit(0)
#         embedding = torch.tensor([])
#         outputs = torch.tensor([])
#         for i in range(num_samples):
#             outs, emb = self.forward(x, Z[i], n_layers)
            
#             emb = emb.unsqueeze(0)
#             outs = outs.unsqueeze(0)
#             if i == 0:
#                 embedding = emb
#                 outputs = outs
#             else:
#                 embedding = torch.cat((embedding, emb), dim=0)
#                 outputs = torch.cat((outputs, outs), dim=0)
#         embedding = torch.mean(embedding, 0)
#         outputs = torch.mean(outputs, 0)
#         return outputs, embedding

#     def fit(self, x, num_samples):
#         """
#         Fits the data with different samples of architectures

#         Parameters
#         ----------
#         x : data
#         num_samples : Number of architectures to sample for KL divergence

#         Returns
#         -------
#         act_vec : Tensor
#             output from different architectures
#         kl_loss: Tensor
#             Kl divergence for each sampled architecture
#         thresholds: numpy array
#             threshold sampled for different architectures
#         """
#         act_vec = []
#         thresholds = []
#         if self.args.vanilla:
#             out = self.vanilla_forward(x)
#             kl_gauss = torch.Tensor([0.]).to(self.device)
#             kl_beta = torch.Tensor([0.]).to(self.device)
#             thresholds.append([0])
#             act_vec.append(out.unsqueeze(0))
#         else:
#             kl = torch.Tensor([0.]).to(self.device)
#             Z, n_layers, thresholds = self.architecture_sampler(num_samples)
            
#             # print(Z, Z.shape)
#             # print(num_samples, thresholds); exit(0)
#             for i in range(num_samples):
#                 out, h = self.forward(x, Z[i], n_layers)
#                 print("hsample", h.shape)
#                 if self.args.bayesian:
#                     kl += self.kl_loss_gauss()
#                 act_vec.append(out.unsqueeze(0))

#             kl_gauss = kl / num_samples
#             kl_beta = self.architecture_sampler.get_kl()

#         act_vec = torch.cat(act_vec, dim=0)
#         exit(0)
#         thresholds = np.asarray(thresholds)
#         return act_vec, kl_beta, thresholds

#     def E_loglike(self, neg_loglike_fun, output, target):
#         """

#         Parameters
#         ----------
#         neg_loglike_fun : Negative log likelihood function
#         output : scores predicted by model
#         target : real labels

#         Returns
#         -------
#         mean negative log likelihood of the model based on different architectures
#         """
#         if self.args.vanilla:
#             num_samples = 1
#         else:
#             num_samples = self.num_samples

#         batch_sze = target.shape[0]
#         print("targetshape")
#         print(target.shape, output.shape)
#         target_expand = target.repeat(num_samples)
        
#         # print(output.shape)
#         output = output.view(num_samples * batch_sze, -1)
#         print(target_expand.shape, output.shape)
#         # exit()
#         neg_loglike = neg_loglike_fun(output, target_expand).view(num_samples, batch_sze)
#         # print(output.shape, target.shape, target_expand.shape, neg_loglike.shape); exit(0)
#         mean_neg_loglike = neg_loglike.mean(0).mean()
#         return mean_neg_loglike