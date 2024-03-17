import timm
import torch
import torch, torch.nn as nn, torch.nn.functional as F
import torch
from torch import nn, einsum
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import math

from model import *

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            LinearWithRegularization(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            LinearWithRegularization(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiheadAttentionWithRegularization(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.5, l2_reg=0.01):
        super(MultiheadAttentionWithRegularization, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.l2_reg = l2_reg

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        output, attn_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # L2 regularization
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)

        # Add L2 regularization term to the attention output
        output = output + self.l2_reg * l2_loss

        return output, attn_weights

class LinearWithRegularization(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=0.01):
        super(LinearWithRegularization, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_decay = weight_decay  # L2 regularization strength

    def forward(self, x):
        # Apply L2 regularization to the linear layer weights
        l2_regularization = self.weight_decay * torch.sum(self.linear.weight ** 2) / 2
        output = self.linear(x) + l2_regularization
        return output

class Transformer(nn.Module):
    def __init__(self, dim=1280, depth=2, heads=4, dim_head=160, mlp_dim=2560, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = self.norm = nn.LayerNorm(dim)
        self.fc_q = LinearWithRegularization(dim, dim)
        self.fc_k = LinearWithRegularization(dim, dim)
        self.fc_v = LinearWithRegularization(dim, dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttentionWithRegularization(embed_dim=dim, num_heads=heads),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, query, key, value):
        base = key
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        for cross_attn, ff in self.layers:
            at, at_weight = cross_attn(query, key, value)
            x = self.norm(at) + base
            x = self.norm(ff(x)) + x
        return x

class Cross_model(nn.Module):
    def __init__(self, num_frames=10, out_dim=3, valid_name='1'):
        super(Cross_model, self).__init__()

        self.num_frames = num_frames
        #######################################
        self.eff = VideoClassifier()
        self.eff.load_state_dict(torch.load(f'./model_saved/Basic_VI_r4_epoch30.pt'))
        self.eff_feature_VI = nn.Sequential(*list(self.eff.children())[:-2])  # 마지막 2개 레이어를 제외한 모든 레이어를 포함

        for param in self.eff_feature_VI.parameters():
            param.requires_grad_(False)

        self.eff_MI = MI_Model()
        self.eff_MI.load_state_dict(torch.load(f'./model_saved/mi_best_valid_1.pth'))

        for param in self.eff_MI.parameters():
            param.requires_grad_(False)
        # self.eff_feature_VI = models.efficientnet_b0(pretrained=True)
        # # self.eff_feature_MI = models.efficientnet_b0(pretrained=True)
        # self.eff_feature_VI.classifier = nn.Sequential()
        # # self.eff_feature_MI.classifier = nn.Sequential()
        #######################################
        #######################################
        #######################################
        max_len = 30
        d_embed = 1280
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        #######################################
        #######################################
        #######################################
        # self.projection_vi = nn.Sequential(LinearWithRegularization(1280, 384))
        # self.projection_mi = nn.Sequential(LinearWithRegularization(1280, 384))

        self.VI_transformer = Transformer()
        self.MI_transformer = Transformer()
        # self.transformer = Transformer()
        self.classifier = nn.Sequential(LinearWithRegularization(1280*2, out_dim))
        # self.classifier = nn.Sequential(LinearWithRegularization(1280*2, 1000),
        # nn.GELU(),
        # nn.Dropout(0.2),
        # LinearWithRegularization(1000,out_dim)
        # )
        # steering, speed, accel

    def forward(self, org_img, optical_img, sensor, device):
        self.eff_feature_VI.eval()
        self.eff_MI.eval()

        b, c, f, h, w = org_img.size()
        org_img = org_img.view(-1, c, h, w)
        org_img = self.eff_feature_VI(org_img)

        # org_img = org_img.view(-1, 1280)
        # org_img = self.projection_vi(org_img)
        org_img = org_img.view(b, f, -1) # batch, frame, feature
        #######################################
        #######################################
        #######################################
        b, c, f, h, w = optical_img.size()
        # optical_img = optical_img.view(-1, c, h, w)
        # optical_img = self.eff_feature_MI(optical_img, sensor)

        optical_img = self.eff_MI.extract_features(optical_img, sensor)

        # b, c, f, h, w = optical_flow.size()

        # optical_img = optical_img.view(-1, 1280)
        # optical_img = self.projection_mi(optical_img)
        optical_img = optical_img.view(b, f, -1) # batch, frame, feature
        #######################################
        #######################################
        #######################################
        org_img = org_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
        optical_img = optical_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
        #######################################
        #######################################
        VI_feature = self.VI_transformer(optical_img, org_img, org_img) # batch, frame, feature
        MI_feature = self.MI_transformer(org_img, optical_img, optical_img) # batch, frame, feature
        x_feature = torch.cat([VI_feature,MI_feature],dim=2) # # batch, frame, feature*2
        # print(x_feature.shape)
        x_feature = torch.mean(x_feature[:, :, :], axis=1) # batch, feature*2
        x_feature = self.classifier(x_feature)
        # x_feature = self.sigmoid(x_feature)
        return x_feature


class CrossAttention(nn.Module):
    def __init__(self, feature_size, num_heads=1):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads)

    def forward(self, query, key, value):
        # query, key, value가 이미 배치 차원을 포함한 3D 텐서인 경우
        # 직접 멀티헤드 어텐션 함수에 전달
        out, _ = self.attention(query, key, value)
        return out

class DriverClassifier(nn.Module):
    def __init__(self, video_feature_extractor, sensor_feature_extractor, feature_size=1280, num_classes=3):
        super(DriverClassifier, self).__init__()
        self.video_feature_extractor = video_feature_extractor
        self.sensor_feature_extractor = sensor_feature_extractor
        self.cross_attention = CrossAttention(feature_size=feature_size)
        self.final_classifier = nn.Linear(feature_size, num_classes)

    def forward(self, video_input, optical, sensor_input):
        # 비디오 및 센서 특성 추출
        b, c, f, h, w = video_input.size()
        video_input = video_input.view(-1, c, h, w)
        video_features = self.video_feature_extractor(video_input)
        video_features = video_features.view(b, f, -1) # batch, frame, feature
        b, c, f, h, w = optical.size()
        sensor_features = self.sensor_feature_extractor.extract_features(optical, sensor_input)
        sensor_features = sensor_features.view(b, f, -1) # batch, frame, feature

        # 비디오 특성을 key 및 value로, 센서 특성을 query로 사용하여
        # CrossAttention 모듈을 통해 특성 결합
        combined_features = self.cross_attention(sensor_features, video_features, video_features)

        # 결합된 특성을 최종 분류기에 전달
        final_output = self.final_classifier(combined_features)
        final_output = torch.mean(final_output, dim=1)
        return final_output