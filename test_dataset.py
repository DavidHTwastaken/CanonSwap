import os
import os.path as osp
from os.path import basename
import numpy as np
import pandas as pd
import torch
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.can_swap_pipeline_e2e import CanSwapPipeline
from inference_canswap import partial_fields, fast_check_ffmpeg, fast_check_args
from src.can_swap_pipeline_e2e import CanSwapPipeline
from tqdm import tqdm

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )


def setup_pipeline(args: ArgumentConfig):
    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    inference_cfg.flag_stitching = False
    inference_cfg.flag_pasteback = True
    inference_cfg.flag_use_half_precision = False

    canswap_pipeline = CanSwapPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )
    return canswap_pipeline

def same_gender(vid_name: str, img_name: str, image_csv: pd.DataFrame):
    # vid is from RAVDESS, even actor number is female
    vid_is_female = int(vid_name.split('-')[-1].split('.')[0]) % 2 == 0
    # img has a CSV file showing sex of the subject
    img_is_female = image_csv.loc[image_csv['filename'] == img_name, 'sex'].values[0] == 'female'
    # print(vid_name, img_name, 'skipped' if vid_is_female != img_is_female else '')
    return vid_is_female == img_is_female


root = os.path.join('..', 'diverse-face-dataset')
output_dir = 'results'

# m = pd.read_csv(os.path.join(root, 'map.csv'), header=0)
# print(m)
vid_data_dir = os.path.join(root, 'targets')
videos = list(sorted(os.path.join(vid_data_dir, v)
              for v in os.listdir(vid_data_dir) if v.endswith('.mp4')))
img_data_dir = os.path.join(root, 'sources')
images = list(sorted(os.path.join(img_data_dir, i)
              for i in os.listdir(img_data_dir) if i.lower().endswith('jpg')))
image_csv = pd.read_csv(os.path.join(img_data_dir, 'identities.csv'))
# # preprocess each video and image
# videos = m['file'][m['is_video'] == 1]
# images = m['file'][m['is_video'] == 0]

# run infer.py on each image with each video; save results in outputs/{image}_{video}
cropped_videos = []
cropped_images = []
canswap_pipeline = setup_pipeline(ArgumentConfig())
for img in tqdm(images):
    for v in tqdm(videos, desc=f"Running all videos for {img}"):
        if not same_gender(os.path.basename(v), os.path.basename(img), image_csv):
            continue
        # skip already-completed combinations
        save_filename = f'{osp.splitext(basename(v))[0]}--{osp.splitext(basename(img))[0]}.mp4'
        if osp.exists(osp.join(output_dir, save_filename)):
            print(f'Skipping {img} with {v}')
            continue
        args = ArgumentConfig(
            source=os.path.join(img),
            driving=os.path.join(v)
        )
        # subprocess.run(["python", "infer.py", 'examples', "--source", os.path.join(img), "--target", os.path.join(v), "--output", f'{img.split(".")[0]}_{v.split(".")[0]}'])
        print(f'Processing {img} with {v}')
        fast_check_args(args)
        canswap_pipeline.execute(args)
