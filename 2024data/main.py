import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import cv2
import os
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvideotransforms import video_transforms, volume_transforms
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import natsort

warnings.filterwarnings("ignore", category=FutureWarning)

from model import *
from cross_attention import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 5, 6"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 서버적용 코드
device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

# class종류
# classes = {'leegihun': 0, 'leeyunguel': 1, 'leeseunglee': 2}
classes = {'leegihun': 0, 'leeyunguel': 1}
validation_round = 6


def get_transforms(img_size=224):
    transforms_vi = video_transforms.Compose([
        video_transforms.Resize(224),
        # video_transforms.RandomCrop((224, 224)),
        video_transforms.RandomHorizontalFlip(),
        # video_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    transforms_mi = video_transforms.Compose([
        video_transforms.Resize(224),
        # video_transforms.RandomCrop((224, 224)),
        video_transforms.RandomHorizontalFlip(),
        # video_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transforms_val = video_transforms.Compose([
        video_transforms.Resize(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    return transforms_vi, transforms_mi, transforms_val


# class Cross_Dataset(Dataset):
#     def __init__(self, video_paths, transform=None):
#         self.video_paths = video_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.video_paths)

#     def __getitem__(self, idx):
#         ## video ##
#         video = self.load_video(self.video_paths[idx])
#         if self.transform:
#             video_tensor = self.transform(video)

#         ## Optical flow ##
#         optical_path = self.video_paths[idx].replace('split_data', 'optical_data')
#         video_name = self.video_paths[idx].split('/')[-1].split('.')[0]
#         optical_base_path = os.path.dirname(optical_path)
#         search_image = os.path.join(optical_base_path, f'{video_name}_*.jpg')
#         image_paths = sorted(glob.glob(search_image))
#         optical_flow = self.load_optical(image_paths)

#         if self.transform:
#             optical_flow_tensor = self.transform(optical_flow)

#         ## Sensor ##
#         sensor_path = self.video_paths[idx].replace('video', 'sensor')
#         sensor_base_path = os.path.dirname(sensor_path)
#         sensor_file_path = os.path.join(sensor_base_path, f'{video_name}.csv')
#         sensor_data = pd.read_csv(sensor_file_path)
#         sensor_data_resampled = self.resample_sensor_data(sensor_data, len(optical_flow))
#         sensor_data_tensor = torch.tensor(sensor_data_resampled.values, dtype=torch.float32)

#         ## label ##
#         id = self.video_paths[idx].split('/')[2]
#         # print(self.video_paths[idx])
#         label = classes[id]

#         return video_tensor, optical_flow_tensor, sensor_data_tensor, label

#     def load_video(self, video_path):
#         # 비디오 파일을 로드하고 전처리하는 함수
#         cap = cv2.VideoCapture(video_path)
#         frames = []
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         selected_frames = np.linspace(0, frame_count - 1, 10, dtype=int)

#         for i in range(frame_count):
#             ret, frame = cap.read()
#             if i in selected_frames:
#                 frames.append(frame)

#         cap.release()

#         if len(frames) == 0:
#             raise ValueError(f"No frames extracted for video: {video_path}")

#         frames = np.stack(frames, axis=0)  # 프레임을 하나의 배열로 결합
#         return frames

#     def load_optical(self, image_paths):
#         opticals = []
#         optical_count = len(image_paths)
#         selected_optical = np.linspace(0, optical_count - 1, 10, dtype=int)

#         for i in range(optical_count):
#             if i in selected_optical:
#                 image = Image.open(image_paths[i])
#                 opticals.append(np.array(image))

#         optical = np.stack(opticals, axis=0)  # optical을 하나의 배열로 결합

#         return optical

#     def resample_sensor_data(self, sensor_data, num_images):
#         # 센서 데이터 중 x축 데이터만 선택
#         x_axis_data = sensor_data['x']  # 'x'는 x축 데이터를 나타내는 컬럼 이름입니다. 실제 데이터에 맞게 조정해주세요.

#         # x축 데이터를 옵티컬 플로우 이미지 수에 맞춰 재샘플링
#         intervals = np.array_split(x_axis_data, num_images)
#         resampled = [interval.mean() for interval in intervals]
#         resampled_data = pd.DataFrame(resampled, columns=['x'])  # 재샘플링된 데이터를 DataFrame으로 변환

#         return resampled_data

class Cross_Dataset(Dataset):
    def __init__(self, visual_paths, optical_paths, transform=None):
        self.visual_paths = visual_paths
        self.optical_paths = optical_paths
        self.transform = transform

    def __len__(self):
        return len(self.visual_paths)

    def __getitem__(self, idx):
        visual_lt = []
        optical_lt = []

        ## Video ##
        visual_path = self.visual_paths[idx]
        search_visual_image = natsort.natsorted(glob.glob(f'{visual_path}_mean_*.jpg'))

        ## Optical flow ##
        optical_path = self.optical_paths[idx]
        search_optical_image = natsort.natsorted(glob.glob(f'{optical_path}_mean_*.jpg'))

        for i in range(len(search_visual_image)):
            visual = np.array(Image.open(search_visual_image[i]))
            visual_lt.append(visual)

            optical = np.array(Image.open(search_optical_image[i]))
            optical_lt.append(optical)

        if self.transform:
            visual_tensor = self.transform(visual_lt)
            optical_tensor = self.transform(optical_lt)


        ## Sensor ##
        # visual_path 기준으로 영상 파일명 가져와 csv 불러오기
        sensor_path = visual_path.replace('video', 'sensor')
        sensor_path = sensor_path.replace('visual_window_30', 'split_data')
        search_sensor_file = f'{sensor_path}.csv'
        sensor_data = pd.read_csv(search_sensor_file)

        num_images = len(optical_lt)
        # 센서 데이터 중 x축 데이터만 선택
        x_axis_data = sensor_data['x']  # 'x'는 x축 데이터를 나타내는 컬럼 이름입니다. 실제 데이터에 맞게 조정해주세요.

        # x축 데이터를 옵티컬 플로우 이미지 수에 맞춰 재샘플링
        intervals = np.array_split(x_axis_data, num_images)
        resampled = [interval.mean() for interval in intervals]
        resampled_data = pd.DataFrame(resampled, columns=['x'])  # 재샘플링된 데이터를 DataFrame으로 변환
        sensor_tensor = torch.tensor(resampled_data.values, dtype=torch.float32) # tensor 변환

        ## label ##
        # visual_path 기준으로 영상 파일명 가져와 label 지정
        id = visual_path.split('\\')[-1].split('_')[0]
        label = classes[id]

        return visual_tensor, optical_tensor, sensor_tensor, label


### Dataset ###
transforms_vi, transforms_mi, transforms_val = get_transforms()

# visual 파일 경로 추출
visual_file_paths = glob.glob('video/visual_window_30/*/*/*/*/*')
unique_path = set()
for filename in visual_file_paths:
    prefix = filename.split("_mean_")[0]
    unique_path.add(prefix)
visual_file_paths = natsort.natsorted(list(unique_path))

# optical 파일 경로 추출
optical_file_paths = glob.glob('video/optical_window_30/*/*/*/*/*')
unique_path = set()
for filename in optical_file_paths:
    prefix = filename.split("_mean_")[0]
    unique_path.add(prefix)
optical_file_paths = natsort.natsorted(list(unique_path))

train_visual_file_paths = natsort.natsorted([path for path in visual_file_paths if not glob.fnmatch.fnmatch(path, f'video/visual_window_30/*/*/*/{validation_round}/*')])
train_optical_file_paths = natsort.natsorted([path for path in optical_file_paths if not glob.fnmatch.fnmatch(path, f'video/optical_window_30/*/*/*/{validation_round}/*')])

valid_visual_file_paths = natsort.natsorted([path for path in visual_file_paths if glob.fnmatch.fnmatch(path, f'video/visual_window_30/*/*/*/{validation_round}/*')])
valid_optical_file_paths = natsort.natsorted([path for path in optical_file_paths if glob.fnmatch.fnmatch(path, f'video/optical_window_30/*/*/*/{validation_round}/*')])

### Dataset ###
train_dataset = Cross_Dataset(train_visual_file_paths, train_optical_file_paths, transforms_mi)
valid_dataset = Cross_Dataset(valid_visual_file_paths, valid_optical_file_paths, transforms_val)

### Dataloader ###
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

## Model ###
model = Cross_model().to(device)
# model = torch.nn.DataParallel(model).to(device)
# vi_model = VideoClassifier().to(device)
# vi_model.load_state_dict(torch.load(f'./model_saved/Basic_VI_r4_epoch30.pt'))
# vi_model_feature = nn.Sequential(*list(vi_model.children())[:-2])
#
# mi_model = MI_Model().to(device)
# mi_model.load_state_dict(torch.load(f'./model_saved/mi_best_valid_1.pth'))
# mi_model_feature = mi_model
#
# vi_model.eval()
# mi_model.eval()
# driver_classifier = DriverClassifier(vi_model_feature, mi_model_feature, feature_size=1280, num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

### Train ###
num_epochs = 20
best_val_loss = float('inf')
best_val_accuracy = 0.0

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss, total_accuracy = 0, 0

    for video_data, optical_data, sensor_data, targets in tqdm(train_loader):
        video_data, optical_data, sensor_data, targets = video_data.to(device), optical_data.to(device), sensor_data.to(
            device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(video_data, optical_data, sensor_data, device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        total_accuracy += correct / total

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)

    ## Validation ##
    model.eval()
    total_val_loss, total_val_accuracy = 0, 0

    with torch.no_grad():
        for video_data, optical_data, sensor_data, targets in tqdm(valid_loader):
            video_data, optical_data, sensor_data, targets = video_data.to(device), optical_data.to(
                device), sensor_data.to(device), targets.to(device)
            outputs = model(video_data, optical_data, sensor_data, device)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            total_val_accuracy += correct / total

    avg_val_loss = total_val_loss / len(valid_loader)
    avg_val_accuracy = total_val_accuracy / len(valid_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')

    # validation 기준으로 모델 저장
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), f'model_saved/cross_best_valid_{epoch}.pth')  # 모델 저장
        print(f"Best model saved to 'cross_best_valid_{epoch}.pth'.")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()