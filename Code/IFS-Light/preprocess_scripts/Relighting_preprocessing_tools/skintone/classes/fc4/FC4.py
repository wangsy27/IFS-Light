from typing import Union

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from auxiliary.settings import USE_CONFIDENCE_WEIGHTED_POOLING, DEVICE
#from classes.fc4.repvit.repvitLoader import RepViTLoader
from classes.fc4.repvit.repvit import repvit_m0_9, repvit_m2_3, repvit_m0_6
#from classes.fc4.repvit.Repvitattention import repvit_m0_6, repvit_m0_9
from classes.fc4.repvit.utils import replace_batchnorm
from timm.models import create_model
import numpy as np
import timm
from torch import nn, einsum
from einops import rearrange, repeat
import math
from inspect import isfunction



class FC4(torch.nn.Module):

    def __init__(self, img: Tensor=None):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        #squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=True)
        #self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])

        #repvit
        #RepViT = RepViTLoader(1.1).load(pretrained=True)
        repvit = repvit_m0_6(False, 1000, False)
        #repvit = repvit_m2_3(False, 1000, True)
        # checkpoint = torch.load('/home/wsy/fc4-pytorch/classes/fc4/repvit/repvit_m0_9_distill_450e.pth', map_location=DEVICE)
        # repvit.load_state_dict(checkpoint['model'])
        # repvit.eval()
        
        self.backbone = repvit
        #self.backbone2 = repvit

        # self.fc = nn.Sequential(
        #     nn.Linear(320, 160),
        #     nn.Sigmoid(),
        #     nn.Linear(160, 3),
        #     nn.Sigmoid()
        # )

        self.skin_branch = nn.Sequential(
            nn.Linear(1000, 128),
            nn.Sigmoid(),
            nn.Linear(128, 3)  # 输出 3 个值（R, G, B）
        )

        # 光色分支（回归任务，输出 SH 系数）
        self.light_branch = nn.Sequential(
            nn.Linear(1000, 256),
            nn.Sigmoid(),
            nn.Linear(256, 27)  # 输出 27 个 SH 系数
        )


    

    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """
        x = self.backbone(x)
        #y = self.backbone2(y)
        
        # mix = torch.cat((x,y), 1)
        # out = self.fc(mix)
        # if len(x[0]) == 24:
        #     x = x[0]
        #     x = x.view(x.size(0), x.size(1), 1, 1)
        # else:
        #     x = x.view(x.size(0), x.size(1), 1, 1)
        #x = x.view(x.size(0), x.size(1), 1, 1)
        
        skin_output = self.skin_branch(x)
        light_output = self.light_branch(x)

        return skin_output, light_output
        

        # if len(c[0]) == 24:
        #     c = c[0]
        #     c= c.view(c.size()[0]//3, c.size()[1]*3, 1, 1)
        # else:
        #     c = c.view(c.size()[0]//3, c.size()[1]*3, 1, 1)
        #out2 = self.backbone2(y)
        #c = torch.cat((out1, out2),0)
        #c = c.view(c.size()[0]//6, c.size()[1]*6, 1, 1)
        #out = self.fc(c)
        # if out.numel() == 3:
        #     out = out.unsqueeze(0)
        # out = self.final_convs(x)
        

        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps
        # if USE_CONFIDENCE_WEIGHTED_POOLING:
        #     # Per-patch color estimates (first 3 dimensions)
        #     rgb = normalize(out[:, :3, :, :], dim=1)

        #     # Confidence (last dimension)
        #     confidence = out[:, 3:4, :, :]

        #     # Confidence-weighted pooling
        #     pred = normalize(torch.sum(torch.sum(rgb * confidence, 2), 2), dim=1)

        #     return pred, rgb, confidence

        # Summation pooling
        #pred1 = normalize(torch.sum(torch.sum(out, 2), 2), dim=1)
        # pred1 = normalize(out, dim=1)
        # indices = torch.arange(0, pred1.size(0), 3, dtype=torch.long)
        # # pred = (pred1[indices-1]+pred1[indices]+pred1[indices+1])/3.0
        # pred = pred1[indices+1]
        # return pred