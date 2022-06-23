import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
import imageio
from skimage import img_as_ubyte

from frames_dataset import PairedDataset
from logger import Logger, Visualizer


def animate(config, inpainting_network, kp_detector, bg_predictor, dense_motion_network, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animation')
    log_dir = os.path.join(log_dir, 'MEGC2022_epoch79')
    png_dir = os.path.join(log_dir, 'png')

    animate_params = config['animate_params']
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, inpainting_network=inpainting_network, dense_motion_network=dense_motion_network,
                        kp_detector=kp_detector, bg_predictor=bg_predictor)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    inpainting_network.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    if bg_predictor:
        bg_predictor.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video'].cuda()
            source_frame = x['source_video'][:, :, 0, :, :].cuda()

            kp_source = kp_detector(source_frame)

            for frame_idx in range(driving_video.shape[2]):
                driving_frame = driving_video[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                bg_params = None
                if bg_predictor:
                    bg_params = bg_predictor(source_frame, driving_frame)

                dense_motion = dense_motion_network(source_image=source_frame, kp_driving=kp_driving,
                                                    kp_source=kp_source, bg_param=None,
                                                    dropout_flag=False)
                out = inpainting_network(source_frame, dense_motion)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source_frame, driving=driving_frame, out=out)
                visualizations.append(visualization)

            predictions_ = np.concatenate(predictions, axis=1)
            # 保存图片结果
            result_name = '-'.join([x['driving_name'][0], x['source_name'][0]])
            imageio.imsave(os.path.join(png_dir, result_name + '.png'), (255 * predictions_).astype(np.uint8))

            # 保存视频结果
            image_name = result_name + animate_params['format']
            imageio.mimsave(os.path.join(log_dir, image_name), [img_as_ubyte(frame) for frame in predictions], fps=100)
