from email.mime import audio
import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
from skimage.transform import resize
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from augmentation import AllAugmentationTransform
import glob
from functools import partial
import torch


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = mimread(name)
        if len(video[0].shape) == 2:
            video = [gray2rgb(frame) for frame in video]
        if frame_shape is not None:
            video = np.array([resize(frame, frame_shape) for frame in video])
        video = np.array(video)
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, keypoint_dir, au_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.keypoint_dir = keypoint_dir
        self.au_dir = au_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = frame_shape
        # print(self.frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling

        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                # train_videos = os.listdir(os.path.join(root_dir, 'train'))
                # NOTE: 以下只使用三个公开数据集进行训练
                train_videos = []
                for name in os.listdir(os.path.join(root_dir, 'train')):
                    if len(name) < 20:
                        train_videos.append(name)
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)    # 数据增强操作
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            # NOTE: 读取keypoint
            if self.keypoint_dir:
                kp_folder = os.path.join(self.keypoint_dir, 'train' if self.is_train else 'test')
                for name_ in os.listdir(kp_folder):
                    if name in name_:
                        kp = torch.load(os.path.join(kp_folder, name_))
                        if isinstance(kp, np.ndarray):
                            kp = torch.from_numpy(kp)
            # NOTE: 读取AU
            if self.au_dir:
                au_flag = 0
                csv_folder = os.path.join(self.au_dir, 'train' if self.is_train else 'test')
                for name_ in os.listdir(csv_folder):
                    if name in name_:
                        csv_path = os.path.join(csv_folder, name, 'merged_AU.csv')
                        AU_data = pd.read_csv(csv_path)
                        au_flag = 1

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            if self.au_dir and au_flag == 1:
                frame_idx = np.clip(np.sort(np.random.choice(num_frames, replace=True, size=2)), 0, len(AU_data) - 1)
            else:
                frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            if self.frame_shape is not None:
                resize_fn = partial(resize, output_shape=self.frame_shape)
            else:
                resize_fn = img_as_float32

            if type(frames[0]) is bytes:
                video_array = [resize_fn(io.imread(os.path.join(path, frames[idx].decode('utf-8')))) for idx in frame_idx]
            else:
                video_array = [resize_fn(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            # 为什么这里source取0，driving取1？因为训练的时候是直接取整个视频进行训练，并没有特别划分source和driving！
            # len(video_array)=？
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        # if self.is_train:
        #     out['keypoint'] = kp.requires_grad_(True)
        # else:
        #     out['keypoint'] = kp.requires_grad_(False)
        if self.is_train and self.keypoint_dir:
            out['keypoint'] = kp

        if self.is_train and self.au_dir:
            source_AU = AU_data.iloc[frame_idx[0]][:].values.astype(np.float32) / 5.0
            source_AU += np.random.uniform(-0.02, 0.02, source_AU.shape)
            driving_AU = AU_data.iloc[frame_idx[1]][:].values.astype(np.float32) / 5.0
            driving_AU += np.random.uniform(-0.02, 0.02, driving_AU.shape)
            diff_AU = driving_AU - source_AU
            out['AU_diff'] = diff_AU

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, au_dir, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        self.au_dir = au_dir

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        # print(f'pair: {pair}')
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
