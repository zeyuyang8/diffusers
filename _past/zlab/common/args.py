import os
import argparse

from numpy import add
from .settings import SD3_OFFICIAL_DIR

def add_data_args(parser):
    parser.add_argument("-instance_data_dir", type=str)
    parser.add_argument("-instance_prompt", type=str)
    parser.add_argument("-class_prompt", type=str)
    parser.add_argument("-num_class_images", type=int, default=100)
    parser.add_argument("-resolution", type=int, default=512)
    parser.add_argument("-repeats", type=int, default=1)
    parser.add_argument("-center_crop", action="store_true")
    parser.add_argument("-random_flip", action="store_true")

def add_model_args(parser):
    parser.add_argument("-model", type=str, default="sd3.5-medium", choices=["sd3.5-large", "sd3.5-medium"])
    parser.add_argument("-verbose", action="store_true")
    parser.add_argument("-max_sequence_length", type=int, default=77)

def add_training_args(parser):
    parser.add_argument("-train_batch_size", type=int, default=2)
    parser.add_argument("-dataloader_num_workers", type=int, default=0)
    parser.add_argument("-with_prior_preservation", default=False, action="store_true")

def add_logging_args(parser):
    parser.add_argument("-output_dir", type=str)

def configure_args(args):
    if args.model == "sd3.5-large":
        sd3_path = "sd3_large.safetensors"
    elif args.model == "sd3.5-medium":
        sd3_path = "sd3_medium.safetensors"
    args.sd3_official_paths = {
        "clip_l": os.path.join(SD3_OFFICIAL_DIR, "clip_l.safetensors"),
        "clip_g": os.path.join(SD3_OFFICIAL_DIR, "clip_g.safetensors"),
        "t5": os.path.join(SD3_OFFICIAL_DIR, "t5xxl.safetensors"),
        "sd3": os.path.join(SD3_OFFICIAL_DIR, sd3_path),
    }
    return args

def parse_args():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_model_args(parser)
    add_training_args(parser)
    add_logging_args(parser)
    args = parser.parse_args()
    args = configure_args(args)
    
    return args
