import torch
import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def feature_generator(pic, predictor):
    keypoints = np.empty([50, 2], dtype=np.float32)
    # print(pic.shape)
    pic = pic[:, :, 0:3]

    detector = dlib.get_frontal_face_detector()
    dets = detector(pic, 1)
    # print(len(dets))
    if len(dets) != 0:
        for k, d in enumerate(dets):
            shape = predictor(pic, d)
        # print(f'shape: {shape.part(67)}')
        # 68个landmarks中只取50个，编号18-67
        i = 0
        for b in [1, 2, 5, 8, 9, 10, 13, 16, 17]:
            keypoints[i][0] = shape.part(b).x
            keypoints[i][1] = shape.part(b).y
            i += 1
        for b in range(18, 29):
            keypoints[i][0] = shape.part(b).x
            keypoints[i][1] = shape.part(b).y
            i += 1
        for b in range(31, 61):
            keypoints[i][0] = shape.part(b).x
            keypoints[i][1] = shape.part(b).y
            i += 1
        keypoints = torch.tensor(keypoints)
    # print(keypoints.shape)
    
    return keypoints


if __name__ == '__main__':

    # NOTE: dlib_68_landmarks检测模型路径
    seg_model = r'/data/Sirui/MEs-Experiment/GAN_ME/Thin-Plate-Spline-Motion-Model-autodl/shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(seg_model)

    # NOTE: 待提取关键点的数据集路径
    root_dir = r'/data/Sirui/MEs-Experiment/GAN_ME/articulated-animation/data/MEGC2022/train'

    count = 0
    for i in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, i)
        img_path = os.path.join(dir_path, os.listdir(dir_path)[0])  # 只检测首张图片的landmark
        img = (plt.imread(img_path) * 255).astype('uint8')
        
        # 提取到关键点的才进行保存，否则丢弃
        keypoints = feature_generator(img, predictor)
        if keypoints.shape == torch.Size([50, 2]):
            # NOTE: keypoint保存路径
            torch.save(keypoints, os.path.join(r'/data/Sirui/MEs-Experiment/GAN_ME/Thin-Plate-Spline-Motion-Model-autodl/keypoint50new_folder/MEGC2022/train', str(i) + ".pt"))
            count += 1
        print(f'{count} {i} completed.')

    # test
    # dir_path = r'/data/Sirui/MEs-Experiment/GAN_ME/articulated-animation/data/MEGC2022/train/006_1_2'
    # img_path = os.path.join(dir_path, os.listdir(dir_path)[0])  # 只检测首张图片的landmark
    # img = plt.imread(img_path) * 255
    # img = img.astype('uint8')
    # kp = feature_generator(img, predictor)
    # print(kp.shape == torch.Size([50, 2]))
    # torch.save(kp, os.path.join(r'/data/Sirui/MEs-Experiment/GAN_ME/Thin-Plate-Spline-Motion-Model-autodl', "test.pt"))

    # test kp
    # kp_path = r'/data/Sirui/MEs-Experiment/GAN_ME/Thin-Plate-Spline-Motion-Model-autodl/keypoint_folder/MEGC2022/sub0247-06c-02-contempt-01-disgust.pt'
    # kp = torch.load(kp_path)
    # kp = torch.from_numpy(kp)
    # print(type(kp))

    # test au
    # au_path = r'/data/Sirui/MEs-Experiment/GAN_ME/Thin-Plate-Spline-Motion-Model-autodl/au_folder/MEGC2022/train'
    # data_path = r'/data/Sirui/MEs-Experiment/GAN_ME/articulated-animation/data/MEGC2022/train'
    # au_list, data_list = list(), list()
    # for name in os.listdir(au_path):
    #     au_list.append(name)
    # for name in os.listdir(data_path):
    #     data_list.append(name)
    
    # au_set = set(au_list)
    # data_set = set(data_list)
    # print(data_set - au_set)
