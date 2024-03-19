import timm
import torch
import torch, torch.nn as nn, torch.nn.functional as F
import torch
from torch import nn, einsum
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# Visual information(VI)
class VI_Model(nn.Module):
    def __init__(self, enet_type=None, out_dim=3, valid_name='1'):
        super(VI_Model, self).__init__()
        self.enet_name = 'tf_efficientnet_b0_ns'
        self.enet = models.efficientnet_b0(pretrained=True)
        self.enet.classifier = nn.Sequential()
        self.fc = nn.Linear(1280, out_dim)
        self.fc2 = nn.Linear(1280, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, f, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.enet(x)
        x = x.view(b, f, -1)
        x = torch.mean(x, dim=1)
        pred = self.fc(x)
        event = self.fc2(x)
        return pred, self.sig(event)


# class MI_Model(nn.Module):
#     def __init__(self, enet_type=None, out_dim=3, valid_name='1'):
#         super(MI_Model, self).__init__()
#         self.enet = models.efficientnet_b0(pretrained=True)
#         self.enet.classifier = nn.Sequential()  # EfficientNet의 classifier 제거
#         self.fc = nn.Linear(1280, 256)  # EfficientNet 특징을 중간 차원으로 축소
#         self.sensor_fc = nn.Sequential(
#             nn.Linear(1, 128),  # 센서 데이터 입력 차원 (예: x 축)
#             nn.ReLU(),
#             nn.Linear(128, 256)  # 센서 데이터 특징을 위한 출력 차원
#         )
#         self.classifier = nn.Linear(512, 1280)  # 결합된 특징을 처리
#         self.fc2 = nn.Linear(1280, out_dim)  # 최종 출력 차원
#
#     def forward(self, optical_flow, sensor):
#         combined_feature = self.extract_features(optical_flow, sensor)
#         pred = self.fc2(combined_feature)
#         return pred
#
#     def extract_features(self, optical_flow, sensor):
#         b, c, f, h, w = optical_flow.size()
#         optical_flow = optical_flow.view(-1, c, h, w)
#         optical_flow = self.enet(optical_flow)
#         optical_flow = self.fc(optical_flow)
#         optical_flow = optical_flow.view(b, f, -1)
#         optical_feature = torch.mean(optical_flow, dim=1)
#
#         sensor = sensor.view(-1, sensor.size(-1))
#         sensor_feature = self.sensor_fc(sensor)
#         sensor_feature = sensor_feature.view(b, f, -1)
#         sensor_feature = torch.mean(sensor_feature, dim=1)
#
#         combined_feature = torch.cat((optical_feature, sensor_feature), dim=1)
#         combined_feature = self.classifier(combined_feature)
#         return combined_feature
class MI_Model(nn.Module):
    def __init__(self, enet_type=None, out_dim=3, valid_name='1'):
        super(MI_Model, self).__init__()
        self.enet = models.efficientnet_b0(pretrained=True)
        self.enet.classifier = nn.Sequential()
        self.fc = nn.Linear(1280, 1280)
        self.sensor_fc = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1280)
        )
        self.classifier = nn.Linear(2560, 1280)
        self.fc2 = nn.Linear(1280, out_dim)

    def forward(self, optical_flow, sensor):
        b, c, f, h, w = optical_flow.size()
        combined_feature = self.extract_features(optical_flow, sensor)
        combined_feature = combined_feature.view(b, f, -1)
        combined_feature = torch.mean(combined_feature, dim=1)
        pred = self.fc2(combined_feature)
        return pred

    def extract_features(self, optical_flow, sensor):
        b, c, f, h, w = optical_flow.size()
        # 프레임 당 하나의 특징으로 평균내는 대신 전체 특징을 유지
        optical_flow = optical_flow.view(-1, c, h, w)
        print(optical_flow.shape)
        print(sensor.shape)
        optical_flow = self.enet(optical_flow)
        optical_flow = self.fc(optical_flow)

        sensor = sensor.view(-1, sensor.size(-1))
        sensor_feature = self.sensor_fc(sensor)

        # 각 특징을 배치와 프레임 단위로 재구성하지 않고, 바로 결합
        combined_feature = torch.cat((optical_flow, sensor_feature), dim=1)
        combined_feature = self.classifier(combined_feature)

        return combined_feature

# 가현언니 VI pretrain 모델
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(VideoClassifier, self).__init__()
        # EfficientNet-B0를 불러올 때 `weights` 파라미터 사용
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.base_model = efficientnet_b0(weights=weights)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Identity()
        self.feature_extractor = nn.Linear(in_features, 1280)

        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.base_model(x)
        x = self.feature_extractor(x)
        x = x.view(-1, 5, 1280).mean(1)
        x = self.classifier(x)
        return x

