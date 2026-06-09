import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.can_swap_pipeline_e2e import CanSwapPipeline
from inference_canswap import partial_fields, fast_check_ffmpeg, fast_check_args
from src.can_swap_pipeline_e2e import CanSwapPipeline

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )

def run_inference(args: ArgumentConfig):
    fast_check_args(args)

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

    canswap_pipeline.execute(args)


root = os.path.join('..', 'diverse-face-dataset')

m = pd.read_csv(os.path.join(root, 'map.csv'), header=0)
print(m)

# preprocess each video and image
videos = m['file'][m['is_video'] == 1]
images = m['file'][m['is_video'] == 0]

# run infer.py on each image with each video; save results in outputs/{image}_{video}
cropped_videos = []
cropped_images = []
for img in images:
    for v in videos:
        args = ArgumentConfig(
            source=os.path.join(root, img),
            driving=os.path.join(root, v)
        )
        # subprocess.run(["python", "infer.py", 'examples', "--source", os.path.join(img), "--target", os.path.join(v), "--output", f'{img.split(".")[0]}_{v.split(".")[0]}'])
        run_inference(args)
