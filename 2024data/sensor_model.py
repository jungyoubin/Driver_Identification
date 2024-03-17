import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import default_collate


device = "cuda" if torch.cuda.is_available() else "cpu"

class SensorDataset(Dataset):
    def __init__(self, file_paths, labels, window_size, step_size):
        self.file_paths = file_paths
        self.labels = labels
        self.window_size = window_size
        self.step_size = step_size

        self.data = []
        self.targets = []
        self.min_val, self.max_val = self.compute_global_min_max(file_paths)

        for path, label in zip(file_paths, labels):
            window_data = []

            # 센서 데이터 로드
            df = pd.read_csv(path, usecols=['x', 'y', 'z'])
            data_normalized = self.min_max_normalize(df, self.min_val, self.max_val)
            # statistical_features = self.extract_statistical_features(data_normalized)

            # 슬라이딩 윈도우 적용 및 통계적 특성 추출
            for start in range(0, data_normalized.shape[0] - window_size + 1, step_size):
                window = data_normalized.iloc[start:start + window_size]
                statistical_features = self.extract_statistical_features(window)
                window_data.append(statistical_features)

            data = np.array(window_data)
            self.data.append(data)
            self.targets.append(label)

    def extract_statistical_features(self, df):
        # DataFrame의 값으로부터 평균, 표준편차, 최대값, 최소값 계산
        mean = df.mean(axis=0).values
        std = df.std(axis=0).values

        # 통계적 특성을 하나의 배열로 결합
        features = np.concatenate([mean, std], axis=0)
        return features

    def compute_global_min_max(self, file_paths):
        # 전체 데이터셋의 최소값과 최대값을 계산하기 위한 리스트
        all_data = []
        for path in file_paths:
            df = pd.read_csv(path, usecols=['x', 'y', 'z'])
            all_data.append(df.values)
        all_data = np.concatenate(all_data, axis=0)

        # 최소값과 최대값 계산
        min_val = all_data.min(axis=0)
        max_val = all_data.max(axis=0)
        return min_val, max_val

    def min_max_normalize(self, data, min_val, max_val):
        # Min-max 정규화 수행
        return (data - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # PyTorch 모델에 맞게 데이터 타입 변환
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.long)
        return data_tensor, target_tensor


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # GRU 층을 정의합니다. batch_first=True는 입력 텐서의 첫 번째 차원이 배치 크기임을 나타냅니다.
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # GRU의 마지막 은닉 상태를 사용하여 클래스 예측을 위한 선형 층을 정의합니다.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x의 형태가 [배치 크기, 시퀀스 길이, 특징 수]인지 확인합니다.
        # 배치 크기를 얻습니다.
        batch_size = x.size(0)

        # GRU의 초기 은닉 상태를 생성합니다.
        # 이 때, 배치 크기와 은닉 상태 크기에 맞는 형태로 초기화해야 합니다.
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # GRU를 통과시킵니다. GRU는 LSTM과 달리 셀 상태(c0)가 없습니다.
        out, _ = self.gru(x, h0)

        # 시퀀스의 마지막 요소에서 클래스 예측을 위한 선형 층을 통과시킵니다.
        out = self.fc(out[:, -1, :])
        return out

def custom_collate_fn(batch):
    # 데이터 샘플의 크기를 확인하고, 모든 샘플을 동일한 크기로 조정
    batch = [item for item in batch if item is not None]  # None 항목 제거
    min_length = min(x[0].shape[0] for x in batch)
    batch = [(x[0][:min_length, :], x[1]) for x in batch]  # 모든 샘플을 최소 길이로 잘라냄
    return default_collate(batch)  # 수정된 배치를 default_collate 함수로 전달

### Dataset ###
file_paths = glob.glob('sensor/split_data/*/*/*/*/*')
class_to_label = {}
label_counter = 0

labels = []
for path in file_paths:
    class_name = path.split('\\')[1]

    if class_name not in class_to_label:
        class_to_label[class_name] = label_counter
        label_counter += 1

    labels.append(class_to_label[class_name])

dataset = SensorDataset(file_paths, labels, 5, 5)
train_dataset, valid_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


### Dataloader ###
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)


## Model ###
input_size = dataset[0][0].shape[1]  # 특성(column)의 수
hidden_size = 128  # LSTM 은닉 상태의 크기
num_layers = 2  # LSTM 층의 수
num_classes = len(np.unique(labels))  # 분류할 클래스의 수

model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


### Train ###
num_epochs = 500
best_val_loss = float('inf')
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()  # 훈련 모드
    total_loss, total_accuracy = 0, 0

    for data, targets in tqdm(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        total_accuracy += correct / total

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)

    # Validation
    model.eval()  # 평가 모드
    total_val_loss, total_val_accuracy = 0, 0

    with torch.no_grad():
        for data, targets in tqdm(valid_loader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            total_val_accuracy += correct / total

    avg_val_loss = total_val_loss / len(valid_loader)
    avg_val_accuracy = total_val_accuracy / len(valid_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')

    # validation 기준으로 모델 저장
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')  # 모델 저장

print("Training complete. Best model saved to 'best_model.pth'.")