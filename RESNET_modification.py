'''
author:zhaoyli
date:2020/4/30
Purpose of the program:修改resnet 50 第五层5_x（对应layer 4） 第一个block 第二个卷积步长为1，这样可以保持第四层（对应layer 3）
的输出图片size不变，更好地提取特征，目前很多论文都是这样做的。

经过验证 跟https://blog.csdn.net/qq_37405118/article/details/105847809 中的 图片一样。
'''

import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

numclasses = 751
class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()


        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,

        )
        layer_5 = nn.Sequential(
            Bottleneck(1024, 512,#关键所在，步长为1
                       downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))

        self.p1 = nn.Sequential(copy.deepcopy(layer_5))

    def forward(self, x):
        x = self.backbone(x)
        x = self.p1(x)

        return x


class EXTRACT_F(nn.Module):
    def __init__(self):
        super(EXTRACT_F,self).__init__()

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(2048, 256, (1, 1))
        self.p2 = nn.AvgPool2d((10,10))
        self.p3 = nn.AvgPool2d((5,10))
        self.conv2 = nn.Conv2d(2048,256,(1,1))
        self.fc = nn.Linear(256,numclasses)
    def forward(self, x):
        x1 = self.GAP(x)  # 1 0248 1 1
        g_f = self.conv(x1).squeeze(3).squeeze(2)#1 256
        g_f = self.fc(g_f)


        p2_f = self.p2(x)# [1 2048 3 1]

        p2_f_1 = p2_f[:,:,0:1,:]# [1 2048 1 1]
        p2_1 = self.conv2(p2_f_1).squeeze(3).squeeze(2)# [1 256 ]
        p21 = self.fc(p2_1)

        p2_f_2 = p2_f[:,:,1:2,:]
        p2_2 = self.conv2(p2_f_2).squeeze(3).squeeze(2)
        p22  = self.fc(p2_2)

        p2_f_3 = p2_f[:,:,2:3,:]
        p2_3 = self.conv2(p2_f_3).squeeze(3).squeeze(2)
        p23 = self.fc(p2_3)


        p3_f = self.p3(x)# [1 2048 6 1]

        p3_f_1 = p3_f[:,:,0:1,:]# [1 2048 1 1]
        p3_1 = self.conv2(p3_f_1).squeeze(3).squeeze(2)# [1 256 ]
        p31 = self.fc(p3_1)

        p3_f_2 = p3_f[:,:,1:2,:]
        p3_2 = self.conv2(p3_f_2).squeeze(3).squeeze(2)
        p32 = self.fc(p3_2)

        p3_f_3 = p3_f[:,:,2:3,:]
        p3_3 = self.conv2(p3_f_3).squeeze(3).squeeze(2)
        p33 = self.fc(p3_3)

        p3_f_4 = p3_f[:,:,3:4,:]
        p3_4 = self.conv2(p3_f_4).squeeze(3).squeeze(2)
        p34 = self.fc(p3_4)

        p3_f_5 = p3_f[:,:,4:5,:]
        p3_5 = self.conv2(p3_f_5).squeeze(3).squeeze(2)
        p35 = self.fc(p3_5)

        p3_f_6 = p3_f[:,:,5:6,:]
        p3_6 = self.conv2(p3_f_6).squeeze(3).squeeze(2)
        p36 = self.fc(p3_6)

        predict = torch.cat([g_f,p21,p22,p23,p31,p32,p33,p34,p35,p36],dim= 1)

        return predict,g_f,p21,p22,p23,p31,p32,p33,p34,p35,p36

if __name__ == '__main__':
    net = RESNET()

    part_F = EXTRACT_F()
    x = torch.randn([1,3,480,160])
    T = net(x)

    print("T.SIZE:",T.size())
    predict,g_f,p21,p22,p23,p31,p32,p33,p34,p35,p36 =part_F(T)
    print("g_f size: ",g_f.size(),"\n",'fuse feature size:',predict.size(),'\n','batch part feature size:',p21.size(),p31.size())

