from torch import nn
from torchvision import models
import torch


device = "cuda"


class ResnetCustom(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.model = models.resnet152(pretrained=True).to(device)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, hidden_size, kernel_size=(
            1, 1), stride=(1, 1), bias=False)
        self.gap2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(256, hidden_size, kernel_size=(
            1, 1), stride=(1, 1), bias=False)
        self.gap3 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv4 = nn.Conv2d(512, hidden_size, kernel_size=(
            1, 1), stride=(1, 1), bias=False)
        self.gap4 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(1024, hidden_size, kernel_size=(
            1, 1), stride=(1, 1), bias=False)
        self.gap5 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv7 = nn.Conv2d(2048, hidden_size, kernel_size=(
            1, 1), stride=(1, 1), bias=False)
        self.gap7 = nn.AdaptiveAvgPool2d((1, 1))
        self.grad_cam = False

    def forward(self, img):
        img = img.to(device)  # torch.Size([16, 3, 224, 224])
        modules2 = list(self.model.children())[:-7]
        fix2 = nn.Sequential(*modules2)
        inter_2 = self.conv2(fix2(img))
        v_2 = self.gap2(self.relu(inter_2)).view(-1,
                                                 self.hidden_size).unsqueeze(1)
        modules3 = list(self.model.children())[:-5]
        fix3 = nn.Sequential(*modules3)
        inter_3 = self.conv3(fix3(img))
        v_3 = self.gap3(self.relu(inter_3)).view(-1,
                                                 self.hidden_size).unsqueeze(1)
        modules4 = list(self.model.children())[:-4]
        fix4 = nn.Sequential(*modules4)
        inter_4 = self.conv4(fix4(img))
        v_4 = self.gap4(self.relu(inter_4)).view(-1,
                                                 self.hidden_size).unsqueeze(1)
        modules5 = list(self.model.children())[:-3]
        fix5 = nn.Sequential(*modules5)
        inter_5 = self.conv5(fix5(img))
        v_5 = self.gap5(self.relu(inter_5)).view(-1,
                                                 self.hidden_size).unsqueeze(1)
        modules7 = list(self.model.children())[:-2]
        fix7 = nn.Sequential(*modules7)  # torch.Size([16, 312, 112, 112])
        o_7 = fix7(img)  # torch.size([16,64,112,112])

        if self.grad_cam:
            self.feat = o_7
        inter_7 = self.conv7(o_7)
        v_7 = self.gap7(self.relu(inter_7)).view(-1,
                                                 self.hidden_size).unsqueeze(1)
        # 维度为 [batch_size, 5, hidden_size]
        return torch.cat((v_2, v_3, v_4, v_5, v_7), 1)

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.feat
