'''
데이터 로더에서 너무 오래 걸려서, 모든 동영상을 불러와서 동영상당 1개의 평균 이미지를 저장하는 코드
'''

import cv2
import os
import glob
import torch
import numpy as np
# from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
# from torchvision import transforms
# from efficientnet_pytorch import EfficientNet
# import torchvision.transforms as transforms
# from torchvision.utils import flow_to_image
# from torchvision import models
# import torch.nn as nn
from PIL import Image

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import cv2
import os
import torch
import natsort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# name_lt = ['baekseungdo', 'cheonaeji', 'chunjihun', 'kimgangsu', 'kimminju', 'leegihun', 'leeseunglee', 'leeyunguel']
name_lt = ['jojeongduk']
course = ['A', 'B', 'C']
event = ['bump', 'corner']

def select_frame(input_video, output_path, set_frame_num):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {path}")

    frames = []
    num=1
    a = 350//2
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (300, 400))
        # frame = frame[a - 112: a + 112, a - 112: a + 112]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # frame[0] = frame[0] + (255 / 300) * num
        # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        frames.append(frame)
        num += 1

    cap.release()
    # frames = np.array(frames[0::3]) # 300 -> 100
    # frames = np.mean(frames, axis=1)
    length = set_frame_num
    frames = frames[::length]
    frame_num = 300 // length
    if len(frames) != frame_num:
        frames.append(frames[-1])

    # 평균 이미지를 파일로 저장
    for i in range(frame_num):
        video_name = input_video.split('\\')[-1].split('.')[0]
        output_file = os.path.join(output_path, f'{video_name}_img_{i}.jpg')

        # print(frames[i])
        # print(output_file)
        # print(average_frame)
        cv2.imwrite(output_file, frames[i])
    print(f"Mean frame saved to {output_file}")

    cap.release()

def mean(input_video, output_path, set_frame_num):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {path}")

    frames = []
    num=1
    a = 350//2
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (350, 350))
        frame = frame[a - 112: a + 112, a - 112: a + 112]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # frame[0] = frame[0] + (255 / 300) * num
        # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
        frames.append(frame)
        num += 1

    cap.release()
    # frames = np.array(frames[0::3]) # 300 -> 100
    # frames = np.mean(frames, axis=1)
    length = len(frames) // set_frame_num
    # frames = [np.mean(frames[f:f + length], axis=0) for f in range(0, len(frames), length)]

    frames = [np.mean(frames[f:f + length], axis=0) for f in range(0, len(frames) - len(frames) % length, length)]

    # 평균 이미지를 파일로 저장
    for i in range(set_frame_num):
        video_name = input_video.split('\\')[-1].split('.')[0]
        output_file = os.path.join(output_path, f'{video_name}_mean_{i}.jpg')

        # print(frames[i])
        # print(output_file)
        # print(average_frame)
        cv2.imwrite(output_file, frames[i])
    print(f"Mean frame saved to {output_file}")

    cap.release()


# def mean_npy(input_video, output_path, video_number):
#
#     # if not cap.isOpened():
#     #     raise FileNotFoundError(f"Unable to open video file: {path}")
#
#
#     num=1
#     # a = 350//2
#     for path in input_video:
#         frames = []
#         # print(path)
#         # frame = cv2.imread(path,cv2.IMREAD_COLOR)
#         for i in range(len(path)):
#             frame = cv2.imread(path[i], cv2.IMREAD_COLOR)
#             # frame = np.array(np.load(path[i]))
#             # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # frame = cv2.resize(frame, (350, 350))
#             # frame = frame[a - 112: a + 112, a - 112: a + 112]
#             # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#             # frame[0] = frame[0] + (255 / 300) * num
#             # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2RGB)
#             frames.append(frame)
#             num += 1
#
#         # cap.release()
#         # frames = np.array(frames[0::3]) # 299 -> 100
#         frames.append(frame) # optical flow image 300개로 맞춰줄려고 한번 더 함
#         # frames = np.mean(frames, axis=1)
#         length = len(frames) // 30
#         # frames = [np.mean(frames[f:f + length], axis=0) for f in range(0, len(frames), length)]
#         frames = [np.mean(frames[f:f + length], axis=0) for f in range(0, len(frames) - len(frames) % length, length)]
#         # print(len(frames))
#         # 평균 이미지를 파일로 저장
#         # video_name = "".join(path[0].split('\\')[-1].split('.')[0]) # name_A_b_1_0of36_0001
#         video_name = "_".join(path[0].split('\\')[-1].split('.')[0].split('_')[:-1])
#
#         for i in range(len(frames)):
#             output_file = os.path.join(output_path, f'{video_name}_mean_{i}.jpg')
#             # print(output_file)
#             # print(frames[i])
#             # print(output_file)
#             # print(average_frame)
#             cv2.imwrite(output_file, frames[i])
#     print(f"Mean frame saved to {output_file}")

def mean_op(input_path, output_path, set_frame_num):
    opticals = []
    for i in input_path:
        optical = np.array(Image.open(i))
        opticals.append(optical)

    length = len(opticals) // set_frame_num
    if length == 0:
        print(input_path[0])
    # frames = [np.mean(frames[f:f + length], axis=0) for f in range(0, len(frames), length)]
    frames = [np.mean(opticals[f:f + length], axis=0) for f in range(0, len(opticals) - len(opticals) % length, length)]
    # print(len(frames))
    # 평균 이미지를 파일로 저장
    # video_name = "".join(path[0].split('\\')[-1].split('.')[0]) # name_A_b_1_0of36_0001
    video_name = "_".join(input_path[0].split('\\')[-1].split('.')[0].split('_')[:-1])

    for i in range(set_frame_num):
        output_file = os.path.join(output_path, f'{video_name}_mean_{i}.jpg')
        # print(output_file)
        # print(frames[i])
        # print(output_file)
        # print(average_frame)
        cv2.imwrite(output_file, frames[i])
    print(f"Mean frame saved to {output_file}")

# ########## visual_image ##########
for n in name_lt:
    for e in event:
        for c in course:
            for i in range(1, 7):
                path = natsort.natsorted(glob.glob(f'./2024data/video/split_data/{n}/{e}/{c}/{i}/*.mp4'))
                # path = f'./data/optical_npy_image/{name}/{c}/{e}/{i}/'  # 본인 경로에 맞게 수정
                # file_list = os.listdir(path)

                for f in path:
                    print(f)
                    # input_video = os.path.join(path, f)  # 본인 경로에 맞게 수정
                    input_video = f
                    output_path = f'./2024data/video/visual_window_10/{n}/{e}/{c}/{i}/'  # 본인 경로에 맞게 수정
                    # print(input_video)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    # print(input_video.split('/')[-1].split('.')[0])
                    mean(input_video, output_path, 10)  # 평균 이미지 생성

            print(f'{n}_{e}_{c}_{i} --> done')

########## optical_image ##########
# for n in name_lt:
#     for e in event:
#         for c in course:
#             for i in range(1, 7):
#                 path_lt = []
#                 path_base = natsort.natsorted(glob.glob(f'./2024data/video/split_data/{n}/{e}/{c}/{i}/*.mp4'))
#                 path = natsort.natsorted(glob.glob(f'./2024data/video/optical_data/{n}/{e}/{c}/{i}/*.jpg'))
#                 num = 299
#                 for b in path_base:
#                     video_name = b.split('\\')[-1].split('.')[0]
#                     optical_path = b.replace('split_data', 'optical_data')
#                     optical_base_path = os.path.dirname(optical_path)
#                     search_optical_image = os.path.join(optical_base_path, f'{video_name}_*.jpg')
#                     # search_optical_image = search_optical_image.replace('\\', '/')
#                     search_optical_image = glob.glob(search_optical_image)
#                     path_lt.append(search_optical_image)
#                 # path = [path[i * num:(i + 1) * num] for i in range((len(path) + num - 1) // num)]
#
#                 for f in path_lt:
#                     output_path = f'./2024data/video/optical_window_10/{n}/{e}/{c}/{i}/'
#                     if not os.path.exists(output_path):
#                         os.makedirs(output_path)
#                     mean_op(f, output_path, 10)
#
#                     print(f'{n}_{e}_{c}_{i} --> done')